# RAG Decision Intelligence

RAG Decision Intelligence is a local question-answering application for a small collection of company policy and handbook documents. It combines dense vector search and BM25 keyword search, merges their rankings, checks retrieved evidence, and generates answers with a local Llama model. The Streamlit interface displays the answer and the retrieved document sections so users can inspect its basis.

The project is a prototype over the Markdown corpus in [`data/raw_docs/`](data/raw_docs/). It is not a policy authority: users should confirm consequential decisions against the source documents.

## Table of contents

1. [Overview](#1-overview)
2. [Key features](#2-key-features)
3. [Technology stack](#3-technology-stack)
4. [Getting started](#4-getting-started)
5. [Usage](#5-usage)
6. [Project structure](#6-project-structure)
7. [Architecture and workflows](#7-architecture-and-workflows)
8. [Results and evaluation](#8-results-and-evaluation)
9. [Limitations](#9-limitations)
10. [Future improvements](#10-future-improvements)
11. [Configuration and troubleshooting](#11-configuration-and-troubleshooting)
12. [Reproducibility and data handling](#12-reproducibility-and-data-handling)

## 1. Overview

Policy information can be spread across multiple documents, and users may describe a concept differently from the document. This project explores how retrieval, evidence assessment, and local generation can work together for policy questions such as benefits, leave, devices, career development, and outside work.

For example, a user might ask about “outside projects” while a source document uses “moonlighting.” The application first retrieves relevant passages, optionally retries with a rewritten query when evidence is weak, then asks the local model to answer from the selected passages. Retrieved source names and sections are included in the interface.

The repository includes 15 Markdown source documents, 95 processed chunks, a prebuilt FAISS index and metadata, an evaluation set of 20 questions, and saved baseline and hybrid evaluation results. The index and processed files are committed; the local GGUF model is excluded by `.gitignore` and must be supplied separately.

> **Screenshot placeholder:** Replace this block with a screenshot of the Streamlit question, answer, and retrieved source display.
>
> `![Application screenshot](docs/images/app-screenshot.png)`

## 2. Key features

- Dense retrieval with `BAAI/bge-base-en-v1.5` and a FAISS inner-product index over normalized vectors.
- BM25 lexical retrieval over each chunk's document name, section, and text.
- Reciprocal Rank Fusion (RRF) using 65% dense rank and 35% BM25 rank, with the RRF constant `k=60`.
- BGE cross-encoder relevance scores used as an evidence signal. The cross-encoder score does not determine the final RRF ordering.
- Evidence labels (`strong`, `moderate`, `weak`) based on the highest combined reranker and metadata keyword-overlap score.
- Corrective retrieval when initial evidence is weak: local query rewriting, a small controlled terminology map, and a second retrieval pass. The corrected results replace the initial results only if the evaluator's score improves.
- Local answer generation with a Llama 3.2 3B Instruct GGUF model through `llama-cpp-python`.
- A Streamlit UI showing the answer, evidence quality, whether correction ran, and source chunks.
- Evaluation scripts that check whether any expected source document appears among the returned top results and record latency.

## 3. Technology stack

| Area | Implementation in this repository |
|---|---|
| Language | Python (no pinned Python version in repository) |
| Embeddings | `sentence-transformers`, model `BAAI/bge-base-en-v1.5` |
| Vector index | `faiss-cpu`; `IndexFlatIP` with normalized embeddings |
| Lexical retrieval | `rank-bm25` / `BM25Okapi` |
| Rank fusion | In-repository weighted Reciprocal Rank Fusion |
| Reranking | `sentence-transformers` `CrossEncoder`, model `BAAI/bge-reranker-base` |
| Local generation | `llama-cpp-python` and a user-provided Llama 3.2 3B Instruct Q4_K_M GGUF file |
| User interface | Streamlit |
| Corpus and artifacts | Markdown and JSON |
| Tests | pytest is used by `test_rag.py` (not declared in `requirements.txt`) |

`requirements.txt` does not pin package versions. The embedding and reranker models are identified by repository code and are fetched by `sentence-transformers` on first use if not already cached. The GGUF model is loaded from a local path and is not downloaded by the application.

## 4. Getting started

### Prerequisites

- Python and pip installed. The repository does not declare a minimum Python version; use a version supported by the listed packages and your platform.
- The repository's checked-in corpus, chunks, FAISS index, and metadata files.
- A local GGUF model at `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`. This file is ignored by Git (`*.gguf`) and is not distributed in the repository. Obtain a compatible model file and place it at that exact path.
- Enough memory and compute for the embedding model, cross-encoder, and local 3B model. The code requests `n_gpu_layers=-1`; hardware/runtime support varies. CPU-only environments may need a compatible `llama-cpp-python` build and code adjustment.
- Network access on first startup if the Hugging Face sentence-transformer models are not cached locally.

No API keys or credentials are configured or required by the checked-in code. The LLM path and model runtime options are constants in `rag/generator.py`; the retrieval model names and artifact paths are constants in `rag/retriever.py`.

### Install

Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pytest` is not listed in `requirements.txt`; install it separately if you want to run the included test:

```bash
python -m pip install pytest
```

`llama-cpp-python` installation can depend on the operating system and available acceleration backend. Use the package's installation guidance for the target machine if the default pip installation is unsuitable.

### Check required local assets

From the repository root, confirm these paths exist before launch:

```text
data/raw_docs/*.md
data/processed/chunks.json
embeddings/index.faiss
embeddings/metadata.json
models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

The model file is intentionally absent from version control in a fresh clone. The committed index and metadata allow normal retrieval without rebuilding, but the application still loads all retrieval models and the local generation model when the pipeline starts.

### Rebuild the corpus artifacts (optional)

If you edit or replace the Markdown corpus, rebuild chunks and the vector index in this order:

```bash
python -m rag.chunker
python embeddings/build_index.py
```

These commands overwrite `data/processed/chunks.json`, `embeddings/index.faiss`, and `embeddings/metadata.json`. Keep all three generated artifacts in sync. BM25 is built in memory from `embeddings/metadata.json` at application startup; it does not have a separate persisted index.

## 5. Usage

### Launch the interface

```bash
streamlit run ui/app.py
```

Enter a question and choose **Ask Question**. The page displays the generated response and retrieval information. Expand a source entry to inspect the document, section, scores when present, and passage text.

### Run a retrieval and answer evaluation

```bash
python evaluation/run_rrf.py
```

The script reads `evaluation/questions.json`, initializes the full RAG pipeline, evaluates every question with `top_k=5`, writes detailed records to `evaluation/rrf_results.json`, and prints summary results. Running it requires the same model and package assets as the UI. It overwrites the saved RRF result file.

The baseline runner is also present:

```bash
python evaluation/run_baseline.py
```

It uses `top_k=8` and writes to `evaluation/baseline_results.json`. The current pipeline implementation is shared; interpret the saved baseline as the output from the repository's prior baseline run, not a guaranteed comparison of separately implemented retrievers.

### Run the included test

```bash
python -m pytest test_rag.py -q
```

This test imports and initializes the RAG pipeline, so it also requires the model file, dependencies, and cached/downloadable retrieval models. The README does not assert a current test result; run the command in the target environment to confirm it there.

### Example questions

Answerable examples represented in the evaluation set include:

- “What type of medical insurance is provided to employees?”
- “Can employees work on outside projects?”
- “What is the company's policy on severance?”

The question “What is the company's annual revenue?” is included as an unsupported example. The generator prompt instructs the model to say when the supplied context does not contain enough information. This is a prompt-level behavior, not a formal guarantee that the model will always abstain correctly.

> **Screenshot placeholder:** Insert a screenshot here showing a question that triggers corrective retrieval and the displayed supporting source.
>
> `![Corrective retrieval example](docs/images/corrective-retrieval.png)`

## 6. Project structure

```text
rag-decision-intelligence/
├── data/
│   ├── raw_docs/                 # Markdown policy and handbook corpus
│   └── processed/chunks.json     # Section-aware, overlapping text chunks
├── embeddings/
│   ├── build_index.py            # Create embeddings and FAISS index
│   ├── index.faiss               # Prebuilt dense vector index
│   └── metadata.json             # Chunk text and source metadata
├── evaluation/
│   ├── questions.json            # 20 questions and expected source documents
│   ├── run_baseline.py           # Evaluation runner (top_k=8)
│   ├── run_rrf.py                # Evaluation runner (top_k=5)
│   ├── baseline_results.json     # Saved evaluation output
│   └── rrf_results.json          # Saved evaluation output
├── models/                       # Local model location; GGUF ignored by Git
├── rag/
│   ├── chunker.py                # Markdown cleanup, section split, word chunking
│   ├── evidence.py               # Evidence scores and quality labels
│   ├── generator.py              # Local answer generation and query rewrite
│   ├── loader.py                 # Load Markdown source documents
│   ├── pipeline.py               # Retrieval, correction, evidence, generation flow
│   ├── reranker.py               # Cross-encoder wrapper (separate utility)
│   ├── retriever.py              # Dense, BM25, RRF, and reranker integration
│   └── retriever_score_fusion.py # Additional/older retrieval implementation
├── ui/app.py                     # Streamlit application
├── test_rag.py                   # Pipeline test
├── manual_*.py                   # Manual component/demo scripts
└── requirements.txt              # Unpinned Python dependencies
```

The active pipeline imports `Retriever` from `rag/retriever.py`. `rag/retriever_score_fusion.py` is present in the repository but is not imported by `rag/pipeline.py`.

## 7. Architecture and workflows

### Components

1. **Document loader (`rag/loader.py`)** reads non-empty `.md` files recursively from `data/raw_docs/` in sorted path order.
2. **Chunker (`rag/chunker.py`)** strips HTML comments, preserves Markdown link text, splits on Markdown headings, and splits section bodies into chunks of up to 350 words with 50-word overlap. It writes chunk IDs, document name/path, section, and text to `data/processed/chunks.json`.
3. **Index builder (`embeddings/build_index.py`)** prepends document and section metadata to each chunk for embedding, encodes with normalized `BAAI/bge-base-en-v1.5` vectors, and writes an inner-product FAISS index plus the chunk metadata JSON.
4. **Retriever (`rag/retriever.py`)** encodes a query, searches FAISS, calculates BM25 over document name + section + chunk text, merges up to 15 candidates from each ranking by chunk ID, computes cross-encoder scores, then orders results by weighted RRF.
5. **Evidence evaluator (`rag/evidence.py`)** combines `0.85 × reranker score` and `0.15 × keyword overlap` over query terms versus document name and section. Maximum score at least `0.20` is labeled strong; at least `0.05` is moderate; otherwise weak.
6. **Pipeline (`rag/pipeline.py`)** runs an initial search with at least 15 candidates. Weak evidence triggers the correction flow. It selects corrected results only when their evidence score is greater than the original, trims to requested `top_k`, recalculates evidence, and generates an answer.
7. **Generator (`rag/generator.py`)** uses the local GGUF model for both query rewriting and answer generation. Generation receives the user's original question and selected source chunks.
8. **UI (`ui/app.py`)** caches a single pipeline instance with Streamlit's resource cache and renders answer, correction status, sources, and evidence details.

> **Architecture diagram placeholder:** Add a diagram of the implemented components and artifact paths here.
>
> `![Architecture diagram](docs/images/architecture.png)`

### Workflow A: standard question answering

```text
Question
  -> Dense FAISS search + BM25 search
  -> Merge candidates by chunk ID
  -> Cross-encoder evidence scores
  -> 65/35 weighted RRF ordering
  -> Evidence assessment
  -> Select top-k passages
  -> Local Llama answer using those passages
  -> UI displays answer and source chunks
```

The reranker contributes to evidence assessment but does not replace the RRF ordering. The final RRF score is `0.65 / (60 + dense_rank) + 0.35 / (60 + bm25_rank)`, omitting a term when a candidate is absent from that ranking.

### Workflow B: weak-evidence correction

When the initial evidence label is `weak`, the pipeline asks the local model to produce one concise retrieval query. It strips quote and newline characters, applies the first matching controlled phrase mapping (currently variations of “outside project/work” and “side gig” to “moonlighting”), and runs retrieval again. If the corrected retrieval has a higher evidence score, it becomes the selected result set. Otherwise, initial results are kept. The displayed `correction_used` field means the retry ran; it does not imply the retry replaced the initial results.

> **Workflow screenshot placeholder:** Add a screenshot of the UI's corrected search query and source evidence here.
>
> `![Correction workflow](docs/images/correction-workflow.png)`

### Workflow C: corpus refresh

```text
Markdown files in data/raw_docs/
  -> python -m rag.chunker
  -> data/processed/chunks.json
  -> python embeddings/build_index.py
  -> embeddings/index.faiss + embeddings/metadata.json
  -> application rebuilds BM25 in memory at startup
```

Changing the corpus without rebuilding its chunks and FAISS index can produce stale or mismatched artifacts. Rebuild both derived stages after corpus edits.

## 8. Results and evaluation

The checked-in `evaluation/questions.json` contains 20 questions: 14 with expected documents and 6 with an empty expected-document list. The evaluator counts a hit when at least one expected document occurs anywhere in the returned source list; it does not measure answer correctness, citation entailment, or whether the exact supporting chunk was returned.

The saved `evaluation/rrf_results.json` records:

| Measure | Saved result |
|---|---:|
| Questions | 20 |
| Questions with expected documents | 14 |
| Expected-document hits | 14 / 14 |
| Coverage on answerable questions | 100% (document-level, top 5) |
| Mean recorded latency | 4.0573 seconds (about 4.06 s) |

These are metrics from the checked-in result file, not a promise of performance on other machines or a newly run evaluation. The latency field wraps `pipeline.answer()` after pipeline initialization; it excludes one-time initialization/model loading. Runtime depends on hardware, model build, cache state, and load. The answerable set is small and project-specific.

The saved baseline output contains 12/14 expected-document hits at top 8 and a recorded mean latency of about 3.53 seconds. The two runs use different `top_k` values and the same current pipeline import, so this repository does not establish a controlled apples-to-apples comparison between independent retrieval configurations.

### Evaluation method

- Inputs: question text, category, and expected document names from `evaluation/questions.json`.
- Retrieval setting: `run_rrf.py` calls `pipeline.answer(question, top_k=5)`; the pipeline requests at least 15 candidates internally.
- Hit rule: at least one expected document name appears among returned sources.
- Aggregate retrieval measure: hit count divided by the number of questions with non-empty expected document lists.
- Timing: `time.perf_counter()` around each call to `pipeline.answer()`; model initialization happens before the loop.
- Output: per-question records, including retrieved documents/sections, correction indicator, rewritten query, generated answer, scores, and latency, saved to `evaluation/rrf_results.json`.
- Not measured: answer factuality, citation precision, abstention accuracy, user satisfaction, statistical uncertainty, or throughput under concurrent load.

## 9. Limitations

- The corpus is small and domain-specific; retrieval quality outside these documents is unknown.
- The 20-question evaluation is small. Its document-level hit metric does not prove answer quality or faithful citations.
- The local 3B model may rewrite queries or answer inconsistently. The prompt's grounding instructions are not a hard enforcement mechanism.
- The controlled terminology map covers only a few phrases and is embedded directly in `rag/pipeline.py`.
- Evidence thresholds and weights are constants and have not been calibrated against a broad, labeled evidence-quality dataset.
- The reranker is used as an evidence signal rather than as the final sort key. This is a project-specific implementation choice.
- The application loads several models locally and can require substantial memory, disk space, and initialization time. The checked-in GGUF file is nearly 2 GB locally but is not tracked or available to a fresh clone.
- `llama-cpp-python` hardware acceleration is platform-specific; the code requests all model layers on GPU and does not expose this setting as configuration.
- Evaluation outputs are committed snapshots. Re-running overwrites them and can produce different answers or timings as environments and dependencies change.
- There is no authentication, authorization, multi-user isolation, telemetry, document version management, or production deployment configuration in the repository.

## 10. Future improvements

- Pin dependencies and document a tested Python/platform matrix.
- Add the optional test dependency to dependency management.
- Move model paths, model IDs, retrieval weights, chunk settings, thresholds, and hardware options into explicit configuration.
- Add a model setup guide with a verified model source and checksum.
- Expand evaluation with labeled relevant chunks, answer faithfulness, citation support, and abstention checks.
- Compare retrieval changes on the same questions, same `top_k`, and repeatable timing conditions.
- Add corpus versioning and a single command that rebuilds chunks, index, and metadata consistently.
- Add metadata filters, document freshness controls, and clearer source citation formatting.
- Consider conversation history only if multi-turn policy questions become a requirement.
- Add operational guidance for privacy, access control, observability, and deployment before using sensitive documents.

## 11. Configuration and troubleshooting

### Current configuration locations

The repository currently configures behavior in source constants rather than environment variables or a config file:

| Setting | Source |
|---|---|
| Raw Markdown directory | `rag/loader.py` (`data/raw_docs`) |
| Chunk output, maximum words, overlap | `rag/chunker.py` (`data/processed/chunks.json`, 350, 50) |
| Embedding model and index paths | `embeddings/build_index.py`, `rag/retriever.py` |
| Reranker model | `rag/retriever.py` (`BAAI/bge-reranker-base`) |
| Fusion weights and constant | `rag/retriever.py` (0.65, 0.35, `k=60`) |
| Evidence weights and thresholds | `rag/evidence.py` (0.85/0.15; 0.20/0.05) |
| Terminology mappings | `rag/pipeline.py` |
| Local GGUF path and inference options | `rag/generator.py` (`n_ctx=4096`, `n_threads=8`, `n_gpu_layers=-1`) |

### Common issues

**`FileNotFoundError` for the GGUF model**

Place the compatible model at `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`. Model files are ignored by Git, so cloning this repository does not provide it.

**FAISS index or metadata cannot be opened**

Run the chunking and index-building commands from the repository root. Confirm `embeddings/index.faiss` and `embeddings/metadata.json` exist and were generated from the same `data/processed/chunks.json`.

**Model download or cache errors on startup**

The embedding and reranker models are loaded by `sentence-transformers`; ensure the machine can reach the model host on first use or has the models cached. The generator is loaded from the local GGUF file and does not use that download path.

**`llama-cpp-python` installation/runtime errors**

Install a wheel/build compatible with your Python, operating system, and hardware. If GPU layers are unavailable, update `n_gpu_layers` in `rag/generator.py` to a supported value (often `0` for CPU inference), understanding this changes runtime behavior.

**The UI exits during initial loading**

Pipeline initialization happens before the question is submitted. Check that dependencies, FAISS artifacts, both transformer models, and the GGUF file are present and loadable. Streamlit's resource cache keeps one initialized pipeline in the running app process.

## 12. Reproducibility and data handling

- Run commands from the repository root unless stated otherwise; artifact paths are relative to the current working directory.
- Dependencies are not version-pinned. Record Python, package, model, and hardware versions when producing evaluation results intended for comparison.
- The evaluation scripts overwrite their corresponding result JSON files. Commit or copy previous outputs before rerunning if you need to preserve a snapshot.
- `data/raw_docs/` contains the project corpus. Review it for permission and privacy before sharing or publishing this repository.
- `.env`, `.env.*`, and `*.gguf` files are ignored by Git. Do not put credentials or private data into tracked files.
- No `LICENSE` file is present in the repository at the time this README was prepared. Add a license and ensure its terms cover the code, corpus, and any model assets before publication; this README does not claim a license.
- No external documentation links are required to understand the checked-in implementation. Confirm external model distribution terms and provenance independently when selecting a GGUF file.
