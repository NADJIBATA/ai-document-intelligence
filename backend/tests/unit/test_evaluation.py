"""RAG evaluation module (repository root ``evaluation/``): metrics, dataset, corpus,
the four systems and the CLI that writes the CSV reports."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import ClassVar

import pytest

from app.rag.embeddings import EmbeddingError, HashingEmbedder
from app.rag.factory import build_analyzer
from app.rag.pipeline import RAGPipeline
from app.rag.reranker import LexicalOverlapReranker
from app.services.ingestion_service import chunk_id_for
from evaluation import run as run_module
from evaluation.corpus import Corpus, document_id_for, load_corpus
from evaluation.dataset import DatasetError, load_dataset, parse_line
from evaluation.metrics import mean, precision_at_k, recall_at_k, reciprocal_rank_at_k
from evaluation.run import SUMMARY_FIELDS, main, parse_ks, parse_systems, resolve_questions
from evaluation.systems import (
    OFFLINE_USER,
    SWITCHES,
    SYSTEMS,
    build_offline_pipeline,
    run_system,
)
from tests.conftest import build_settings

SAMPLE_DIR = Path(__file__).resolve().parents[3] / "evaluation" / "datasets" / "sample"


# ----------------------------------------------------------------------- metrics
class TestMetrics:
    retrieved: ClassVar[list[str]] = ["a", "b", "c", "d", "e"]

    def test_recall(self) -> None:
        assert recall_at_k(self.retrieved, {"b", "z"}, 3) == 0.5
        assert recall_at_k(self.retrieved, {"b", "d"}, 3) == 0.5
        assert recall_at_k(self.retrieved, {"b", "d"}, 5) == 1.0
        assert recall_at_k(self.retrieved, {"z"}, 5) == 0.0

    def test_precision_divides_by_k(self) -> None:
        assert precision_at_k(self.retrieved, {"a", "c"}, 3) == pytest.approx(2 / 3)
        assert precision_at_k(self.retrieved, {"a"}, 5) == 0.2
        # Fewer results than k: still divided by k (missing results are not relevant).
        assert precision_at_k(["a"], {"a"}, 5) == 0.2

    def test_reciprocal_rank(self) -> None:
        assert reciprocal_rank_at_k(self.retrieved, {"a"}, 3) == 1.0
        assert reciprocal_rank_at_k(self.retrieved, {"c", "e"}, 5) == pytest.approx(1 / 3)
        assert reciprocal_rank_at_k(self.retrieved, {"d"}, 3) == 0.0

    def test_duplicates_are_counted_once(self) -> None:
        assert recall_at_k(["a", "a", "b"], {"a", "c"}, 3) == 0.5
        assert precision_at_k(["a", "a", "a"], {"a"}, 3) == pytest.approx(1 / 3)

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(self.retrieved, {"a"}, 0)
        with pytest.raises(ValueError):
            precision_at_k(self.retrieved, set(), 3)

    def test_mean(self) -> None:
        assert mean([1.0, 0.5, 0.0]) == 0.5
        assert mean([]) == 0.0


# ----------------------------------------------------------------------- dataset
class TestDataset:
    def test_parse_line(self) -> None:
        line = json.dumps({"question": " Notice? ", "relevant_chunk_ids": ["x", "y", "x"]})
        parsed = parse_line(line, 4)
        assert parsed is not None
        assert parsed.id == "q4"
        assert parsed.question == "Notice?"
        assert parsed.relevant == ("x", "y")

    def test_blank_lines_and_comments_are_skipped(self) -> None:
        assert parse_line("   ", 1) is None
        assert parse_line("# comment", 2) is None

    @pytest.mark.parametrize(
        "line",
        [
            "{not json",
            "[1, 2]",
            json.dumps({"relevant_chunk_ids": ["a"]}),
            json.dumps({"question": "   ", "relevant_chunk_ids": ["a"]}),
            json.dumps({"question": "q", "relevant_chunk_ids": []}),
            json.dumps({"question": "q", "relevant_chunk_ids": "a"}),
            json.dumps({"question": "q", "relevant_chunk_ids": ["a", 3]}),
        ],
    )
    def test_invalid_lines(self, line: str) -> None:
        with pytest.raises(DatasetError, match="line 7"):
            parse_line(line, 7)

    def test_load_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text(
            "# header\n"
            + json.dumps({"id": "a", "question": "q1", "relevant_chunk_ids": ["x"]})
            + "\n\n"
            + json.dumps({"question": "q2", "relevant_chunk_ids": ["y"]})
            + "\n",
            encoding="utf-8",
        )
        questions = load_dataset(path)
        assert [q.id for q in questions] == ["a", "q4"]

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        line = json.dumps({"id": "a", "question": "q", "relevant_chunk_ids": ["x"]})
        path = tmp_path / "d.jsonl"
        path.write_text(f"{line}\n{line}\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="duplicate id 'a'"):
            load_dataset(path)

    def test_empty_dataset_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "d.jsonl"
        path.write_text("# nothing\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="no question"):
            load_dataset(path)

    def test_sample_dataset_is_valid(self) -> None:
        questions = load_dataset(SAMPLE_DIR / "questions.jsonl")
        corpus = load_corpus(SAMPLE_DIR / "corpus", build_settings())
        resolved = resolve_questions(questions, corpus.resolve)
        assert len(resolved) >= 20
        assert all(question.relevant for question in resolved)


# ------------------------------------------------------------------------ corpus
@pytest.fixture(scope="module")
def sample_corpus() -> Corpus:
    return load_corpus(SAMPLE_DIR / "corpus", build_settings())


class TestCorpus:
    def test_ids_are_deterministic_and_match_the_backend(self, sample_corpus: Corpus) -> None:
        first = sample_corpus.chunks[0]
        assert first.document_id == document_id_for(first.filename)
        assert first.chunk_id == chunk_id_for(first.document_id, first.chunk_index)
        again = load_corpus(SAMPLE_DIR / "corpus", build_settings())
        assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in sample_corpus.chunks]

    def test_resolve(self, sample_corpus: Corpus) -> None:
        chunk = sample_corpus.chunks[1]
        assert sample_corpus.resolve(chunk.ref) == str(chunk.chunk_id)
        assert sample_corpus.resolve(str(chunk.chunk_id)) == str(chunk.chunk_id)
        with pytest.raises(DatasetError, match="unknown chunk"):
            sample_corpus.resolve("missing.txt#0")
        with pytest.raises(DatasetError):
            sample_corpus.resolve(str(uuid.uuid4()))

    def test_unsupported_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        with pytest.raises(ValueError, match=r"image\.png"):
            load_corpus(tmp_path, build_settings())

    def test_empty_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no indexable document"):
            load_corpus(tmp_path, build_settings())


# ----------------------------------------------------------------------- systems
@pytest.fixture(scope="module")
def pipeline(sample_corpus: Corpus) -> RAGPipeline:
    settings = build_settings()
    analyzer = build_analyzer(settings)
    return build_offline_pipeline(
        sample_corpus,
        settings,
        top_k=7,
        embedder=HashingEmbedder(analyzer=analyzer),
        reranker=LexicalOverlapReranker(analyzer),
    )


class TestSystems:
    def test_switches_cover_the_four_systems(self) -> None:
        assert set(SWITCHES) == set(SYSTEMS) == {"bm25", "dense", "hybrid_rrf", "hybrid_rerank"}
        assert SWITCHES["hybrid_rerank"] == {
            "use_lexical": True,
            "use_dense": True,
            "use_reranker": True,
        }

    @pytest.mark.parametrize("system", SYSTEMS)
    def test_every_system_finds_the_relevant_passage(
        self,
        pipeline: RAGPipeline,
        sample_corpus: Corpus,
        system: str,
    ) -> None:
        expected = sample_corpus.resolve("router_x200_manual.txt#2")
        result = run_system(
            pipeline, OFFLINE_USER, system, "factory reset recessed button paperclip", 7
        )
        assert 0 < len(result.chunk_ids) <= 7
        assert expected in result.chunk_ids[:3]
        assert result.degraded == []
        assert result.latency_ms >= 0

    def test_other_users_get_nothing(self, pipeline: RAGPipeline) -> None:
        for system in SYSTEMS:
            result = run_system(pipeline, uuid.uuid4(), system, "factory reset", 5)
            assert result.chunk_ids == []

    def test_unknown_system(self, pipeline: RAGPipeline) -> None:
        with pytest.raises(ValueError, match="unknown system"):
            run_system(pipeline, OFFLINE_USER, "tfidf", "q", 5)


# --------------------------------------------------------------------------- CLI
class TestCli:
    def test_argument_parsers(self) -> None:
        assert parse_ks("7,3,5,3") == (3, 5, 7)
        assert parse_systems("dense,bm25,dense") == ("dense", "bm25")
        for bad in ("0,3", "a", ""):
            with pytest.raises(Exception, match="--k"):
                parse_ks(bad)
        with pytest.raises(Exception, match="--systems"):
            parse_systems("bm25,tfidf")

    def test_offline_run_writes_the_reports(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(run_module, "Settings", build_settings)
        code = main(["--fake-models", "--output-dir", str(tmp_path), "--k", "3,5"])
        assert code == 0

        summaries = list(tmp_path.glob("results_*.csv"))
        details = list(tmp_path.glob("details_*.csv"))
        assert len(summaries) == 1 and len(details) == 1
        with summaries[0].open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert list(rows[0]) == SUMMARY_FIELDS
        assert [(row["system"], row["k"]) for row in rows] == [
            (system, k) for system in SYSTEMS for k in ("3", "5")
        ]
        questions = len(load_dataset(SAMPLE_DIR / "questions.jsonl"))
        for row in rows:
            assert row["questions"] == str(questions)
            assert 0.0 <= float(row["recall"]) <= 1.0
            assert 0.0 <= float(row["precision"]) <= 1.0
            assert 0.0 <= float(row["mrr"]) <= 1.0
            assert row["degraded_queries"] == "0"
        # Recall can only grow with k.
        by_key = {(row["system"], row["k"]): float(row["recall"]) for row in rows}
        for system in SYSTEMS:
            assert by_key[(system, "5")] >= by_key[(system, "3")]

        with details[0].open(encoding="utf-8") as handle:
            detail_rows = list(csv.DictReader(handle))
        assert len(detail_rows) == questions * len(SYSTEMS)
        assert {"recall@3", "precision@5", "rr@5", "retrieved_chunk_ids"} <= set(detail_rows[0])

        output = capsys.readouterr().out
        assert "hybrid_rerank" in output and "Summary:" in output

    def test_unknown_reference_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(run_module, "Settings", build_settings)
        dataset = tmp_path / "bad.jsonl"
        dataset.write_text(
            json.dumps({"id": "x", "question": "q", "relevant_chunk_ids": ["nope.txt#0"]}),
            encoding="utf-8",
        )
        code = main(["--fake-models", "--dataset", str(dataset), "--output-dir", str(tmp_path)])
        assert code == 2
        assert "question 'x'" in capsys.readouterr().err
        assert not list(tmp_path.glob("results_*.csv"))

    def test_live_mode_requires_an_email(self) -> None:
        with pytest.raises(SystemExit):
            main(["--mode", "live"])

    def test_unreachable_embedding_server_gives_a_clear_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def unreachable(*args: object, **kwargs: object) -> RAGPipeline:
            raise EmbeddingError("embeddings: ConnectError")

        monkeypatch.setattr(run_module, "Settings", build_settings)
        monkeypatch.setattr(run_module, "build_offline_pipeline", unreachable)
        assert main(["--output-dir", str(tmp_path)]) == 2
        error = capsys.readouterr().err
        assert "embedding server is unreachable" in error and "--fake-models" in error
        assert not list(tmp_path.glob("results_*.csv"))
