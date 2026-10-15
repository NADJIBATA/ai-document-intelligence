"""Compare the four retrieval systems on a labelled dataset and write CSV reports.

Examples (from the repository root)::

    # Sample corpus + dataset, deterministic fake models (no GPU, no network)
    python -m evaluation.run --fake-models

    # Sample corpus, embedding model and reranker configured in .env (vLLM)
    python -m evaluation.run

    # Data of an existing user of a deployment (PostgreSQL + Qdrant + models)
    python -m evaluation.run --mode live --user-email alice@example.com \\
        --dataset my_questions.jsonl

Outputs, in ``--output-dir``:

* ``results_<timestamp>.csv`` - one row per (system, k): recall, precision, MRR,
  number of questions, mean latency, number of degraded queries;
* ``details_<timestamp>.csv`` - one row per (system, question): rank of the first
  relevant passage, returned passages, per-k metrics.
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.factory import build_analyzer
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import LexicalOverlapReranker
from evaluation.corpus import load_corpus
from evaluation.dataset import DatasetError, EvalQuestion, load_dataset
from evaluation.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank_at_k
from evaluation.systems import OFFLINE_USER, SYSTEMS, build_offline_pipeline, run_system

ROOT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT_DIR / "evaluation" / "datasets" / "sample"
DEFAULT_DATASET = SAMPLE_DIR / "questions.jsonl"
DEFAULT_CORPUS = SAMPLE_DIR / "corpus"
DEFAULT_OUTPUT = ROOT_DIR / "evaluation" / "results"
DEFAULT_KS = (3, 5, 7)

SUMMARY_FIELDS = [
    "system",
    "k",
    "recall",
    "precision",
    "mrr",
    "questions",
    "mean_latency_ms",
    "degraded_queries",
]
DETAIL_FIELDS_PREFIX = ["system", "question_id", "question", "first_relevant_rank"]


@dataclass(frozen=True, slots=True)
class ResolvedQuestion:
    id: str
    question: str
    relevant: frozenset[str]


@dataclass
class SystemReport:
    system: str
    summary: list[dict[str, object]] = field(default_factory=list)
    details: list[dict[str, object]] = field(default_factory=list)


def parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--k expects integers, e.g. 3,5,7") from exc
    if not ks or ks[0] <= 0:
        raise argparse.ArgumentTypeError("--k expects positive integers, e.g. 3,5,7")
    return tuple(ks)


def parse_systems(value: str) -> tuple[str, ...]:
    systems = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    unknown = [system for system in systems if system not in SYSTEMS]
    if not systems or unknown:
        raise argparse.ArgumentTypeError(f"--systems expects a subset of {','.join(SYSTEMS)}")
    return systems


def resolve_questions(
    questions: Sequence[EvalQuestion], resolve: Callable[[str], str]
) -> list[ResolvedQuestion]:
    resolved: list[ResolvedQuestion] = []
    for question in questions:
        try:
            ids = frozenset(resolve(reference) for reference in question.relevant)
        except DatasetError as exc:
            raise DatasetError(f"question '{question.id}': {exc}") from exc
        resolved.append(ResolvedQuestion(question.id, question.question, ids))
    return resolved


def _first_relevant_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> int | None:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return rank
    return None


def evaluate_system(
    pipeline: RAGPipeline,
    user_id: uuid.UUID,
    system: str,
    questions: Sequence[ResolvedQuestion],
    ks: Sequence[int],
) -> SystemReport:
    top_k = max(ks)
    report = SystemReport(system)
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    precisions: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks: dict[int, list[float]] = {k: [] for k in ks}
    latencies: list[float] = []
    degraded = 0

    for question in questions:
        retrieval = run_system(pipeline, user_id, system, question.question, top_k)
        latencies.append(retrieval.latency_ms)
        if retrieval.degraded:
            degraded += 1
        row: dict[str, object] = {
            "system": system,
            "question_id": question.id,
            "question": question.question,
            "first_relevant_rank": _first_relevant_rank(retrieval.chunk_ids, question.relevant)
            or "",
        }
        for k in ks:
            recall = recall_at_k(retrieval.chunk_ids, question.relevant, k)
            precision = precision_at_k(retrieval.chunk_ids, question.relevant, k)
            rr = reciprocal_rank_at_k(retrieval.chunk_ids, question.relevant, k)
            recalls[k].append(recall)
            precisions[k].append(precision)
            reciprocal_ranks[k].append(rr)
            row[f"recall@{k}"] = round(recall, 4)
            row[f"precision@{k}"] = round(precision, 4)
            row[f"rr@{k}"] = round(rr, 4)
        row["latency_ms"] = round(retrieval.latency_ms, 1)
        row["degraded"] = ",".join(retrieval.degraded)
        row["retrieved_chunk_ids"] = " ".join(retrieval.chunk_ids)
        report.details.append(row)

    for k in ks:
        report.summary.append(
            {
                "system": system,
                "k": k,
                "recall": round(mean(recalls[k]), 4),
                "precision": round(mean(precisions[k]), 4),
                "mrr": round(mean(reciprocal_ranks[k]), 4),
                "questions": len(questions),
                "mean_latency_ms": round(mean(latencies), 1),
                "degraded_queries": degraded,
            }
        )
    return report


def detail_fields(ks: Sequence[int]) -> list[str]:
    fields = list(DETAIL_FIELDS_PREFIX)
    for k in ks:
        fields += [f"recall@{k}", f"precision@{k}", f"rr@{k}"]
    return [*fields, "latency_ms", "degraded", "retrieved_chunk_ids"]


def write_reports(
    reports: Sequence[SystemReport], ks: Sequence[int], output_dir: Path, stamp: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"results_{stamp}.csv"
    details_path = output_dir / f"details_{stamp}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for report in reports:
            writer.writerows(report.summary)
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields(ks))
        writer.writeheader()
        for report in reports:
            writer.writerows(report.details)
    return summary_path, details_path


def format_table(reports: Sequence[SystemReport]) -> str:
    header = f"{'system':<15}{'k':>3}{'recall':>9}{'precision':>11}{'mrr':>8}{'latency ms':>12}"
    lines = [header, "-" * len(header)]
    for report in reports:
        for row in report.summary:
            lines.append(
                f"{row['system']!s:<15}{row['k']!s:>3}{row['recall']:>9.3f}"
                f"{row['precision']:>11.3f}{row['mrr']:>8.3f}{row['mean_latency_ms']:>12.1f}"
            )
        degraded = report.summary[0]["degraded_queries"] if report.summary else 0
        if degraded:
            lines.append(f"  ! {report.system}: {degraded} degraded queries (see details)")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run",
        description="Recall / Precision / MRR @k of BM25, dense, hybrid RRF and hybrid+reranker.",
    )
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS, help="offline mode: documents directory"
    )
    parser.add_argument("--user-email", help="live mode: user whose documents are evaluated")
    parser.add_argument("--systems", type=parse_systems, default=SYSTEMS)
    parser.add_argument("--k", dest="ks", type=parse_ks, default=DEFAULT_KS)
    parser.add_argument(
        "--fake-models",
        action="store_true",
        help="offline mode: deterministic hashing embedder and lexical reranker (no server)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "live" and not args.user_email:
        parser.error("--mode live requires --user-email")
    if args.mode == "live" and args.fake_models:
        parser.error("--fake-models is only available in offline mode")

    settings = Settings()
    ks: tuple[int, ...] = args.ks
    top_k = max(ks)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    try:
        dataset = load_dataset(args.dataset)
    except (OSError, DatasetError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    close: Callable[[], None] = lambda: None  # noqa: E731
    try:
        if args.mode == "offline":
            corpus = load_corpus(args.corpus, settings)
            questions = resolve_questions(dataset, corpus.resolve)
            if args.fake_models:
                analyzer = build_analyzer(settings)
                pipeline = build_offline_pipeline(
                    corpus,
                    settings,
                    top_k=top_k,
                    embedder=HashingEmbedder(analyzer=analyzer),
                    reranker=LexicalOverlapReranker(analyzer),
                )
            else:
                pipeline = build_offline_pipeline(corpus, settings, top_k=top_k)
            user_id = OFFLINE_USER
            source = f"{len(corpus.chunks)} passages, {len(corpus.filenames)} documents"
        else:
            from evaluation.live import LiveContext

            context = LiveContext(settings, args.user_email, top_k=top_k)
            close = context.close
            questions = resolve_questions(dataset, context.resolve)
            pipeline = context.retrieval.pipeline
            user_id = context.user_id
            source = f"documents of {args.user_email}"

        sys.stdout.write(
            f"Evaluating {len(questions)} questions ({source}), k={','.join(map(str, ks))}\n\n"
        )
        reports = [
            evaluate_system(pipeline, user_id, system, questions, ks) for system in args.systems
        ]
    except EmbeddingError as exc:
        sys.stderr.write(
            f"error: the embedding server is unreachable at {settings.embedding_base_url} "
            f"({exc}).\nStart it, point EMBEDDING_BASE_URL at it, or use --fake-models.\n"
        )
        return 2
    except (OSError, ValueError) as exc:  # DatasetError is a ValueError
        sys.stderr.write(f"error: {exc}\n")
        return 2
    finally:
        close()

    summary_path, details_path = write_reports(reports, ks, args.output_dir, stamp)
    sys.stdout.write(format_table(reports))
    sys.stdout.write(f"\nSummary: {summary_path}\nDetails: {details_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
