# RAG evaluation

Measures the retrieval quality of the four systems, all run by the production
`RAGPipeline` (only stages are switched off):

| system          | stages                                                     |
|-----------------|------------------------------------------------------------|
| `bm25`          | BM25 only                                                  |
| `dense`         | embeddings + vector search only                            |
| `hybrid_rrf`    | BM25 top 30 + dense top 30, Reciprocal Rank Fusion (k=60)  |
| `hybrid_rerank` | hybrid RRF, then the cross-encoder reranker (production)   |

Metrics, for every k (default 3, 5, 7), averaged over the questions:

* **Recall@k** = relevant passages found in the top k / relevant passages
* **Precision@k** = relevant passages found in the top k / k
* **MRR@k** = mean of 1 / rank of the first relevant passage in the top k (0 if none)

## Dataset

JSON Lines, one question per line (empty lines and `#` comments are ignored):

```json
{"id": "notice", "question": "What is the notice period?", "relevant_chunk_ids": ["service_agreement.txt#3"]}
```

A relevant passage is given by its chunk id (UUID, as returned by `POST /api/search`)
or by `"<file name>#<chunk index>"`. `id` is optional.

## Run (from the repository root, backend dependencies installed)

```bash
# Sample corpus, deterministic fake models: no GPU, no network
python -m evaluation.run --fake-models

# Sample corpus, embedding model and reranker of .env (vLLM servers running)
python -m evaluation.run

# Your own corpus and dataset
python -m evaluation.corpus path/to/corpus          # list passages to label them
python -m evaluation.run --corpus path/to/corpus --dataset path/to/questions.jsonl

# Live: the documents of an existing user (PostgreSQL + Qdrant + models of .env)
python -m evaluation.run --mode live --user-email alice@example.com --dataset my.jsonl

# Options
python -m evaluation.run --k 1,3,5,10 --systems bm25,hybrid_rerank --output-dir /tmp/eval
```

On Windows use `backend\.venv\Scripts\python.exe -m evaluation.run ...`.

Settings are read from the environment and from `.env` in the current directory.
With the Docker stack running (`docker compose up -d`, profile `gpu`), the vLLM servers
are published on the host loopback interface: run the offline evaluation from the host
with `EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1` and
`RERANKER_BASE_URL=http://127.0.0.1:8002` (environment variables override `.env`).
Live mode needs PostgreSQL and Qdrant, which are not published: run it in a one-off
backend container with the `evaluation/` folder mounted:

```bash
docker compose run --rm -v ./evaluation:/app/evaluation backend \
    python -m evaluation.run --mode live --user-email alice@example.com \
    --dataset evaluation/my_questions.jsonl
```

(On a Linux host, add `--user "$(id -u)"` so that the container can write the CSV
files into the mounted folder.)

Offline mode processes the corpus files exactly like uploads (same extraction,
cleaning and chunking settings, same deterministic chunk ids), so `file#index`
references stay valid as long as `CHUNK_SIZE` / `CHUNK_OVERLAP` do not change.
Live mode resolves `file#index` against the indexed documents of that user only.

## Outputs (`evaluation/results/`, not versioned)

* `results_<timestamp>.csv`: `system, k, recall, precision, mrr, questions,
  mean_latency_ms, degraded_queries`
* `details_<timestamp>.csv`: one row per (system, question): rank of the first relevant
  passage, per-k metrics, latency, degraded stages and returned chunk ids.

`degraded_queries` counts the queries where a stage was unavailable (model server
down...). In that case the numbers do not measure the intended system.

## Reading the sample results

The sample corpus (5 documents, 21 passages, 32 questions in English and French) is
small and checks that the tooling works end to end. With `--fake-models` the "dense"
system is a hashing embedder and the reranker a lexical overlap score: their numbers
say nothing about Qwen3-Embedding or bge-reranker-v2-m3. To compare real models, run without
`--fake-models`, ideally on a larger dataset built from your own documents, with
paraphrased questions (BM25 is favoured when questions reuse the document words).
