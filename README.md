# RAG Decision Intelligence

A local Retrieval-Augmented Generation (RAG) system for grounded enterprise-policy question answering.

The system retrieves evidence from a collection of company policy and handbook documents, evaluates retrieval quality, performs corrective retrieval when evidence is weak, and uses a locally hosted Llama model to generate answers grounded only in the retrieved documents.

## Project Highlights

- **15 source documents**
- **95 indexed chunks**
- Hybrid retrieval using **FAISS dense search + BM25 lexical search**
- **65/35 Reciprocal Rank Fusion (RRF)** for combining dense and lexical rankings
- **BGE reranker** used as an evidence signal
- CRAG-style **corrective retrieval** when initial evidence is weak
- Local LLM-based **query rewriting**
- Controlled **domain terminology expansion**
- Local **Llama 3.2 3B Instruct Q4_K_M** generation
- Evidence-aware handling of unsupported questions
- Streamlit user interface
- Automated validation with pytest
- Final evaluation: **14/14 retrieval hits on answerable questions**
- Average evaluation latency: **4.06 seconds**

---

## 1. Problem Statement

Enterprise policy information is often distributed across multiple documents and sections. A simple keyword search may miss relevant passages when users phrase questions differently from the terminology used in the source documents.

For example:

> "Can employees work on outside projects?"

may use the phrase **outside projects**, while the relevant policy document describes the concept as **moonlighting**.

The goal of this project is to build a local question-answering system that can:

1. Retrieve relevant policy evidence.
2. Handle different user phrasings.
3. Detect weak retrieval.
4. Perform corrective retrieval.
5. Generate concise answers grounded in the retrieved evidence.
6. Avoid fabricating information when the documents do not contain an answer.

---

## 2. System Architecture

```text
                    User Question
                          |
                          v
                +--------------------+
                | Initial Retrieval  |
                +--------------------+
                    /            \
                   v              v
             FAISS Dense       BM25
             Retrieval          Search
                   \              /
                    v            v
                +----------------+
                | 65/35 RRF      |
                | Fusion          |
                +-------+--------+
                        |
                        v
                +----------------+
                | BGE Reranker   |
                | Evidence Signal|
                +-------+--------+
                        |
                        v
                 Evidence Check
                  /          \
              Strong          Weak
                |               |
                |               v
                |       Query Rewriting
                |               |
                |               v
                |      Terminology Expansion
                |               |
                |               v
                |      Corrective Retrieval
                |               |
                +-------+-------+
                        |
                        v
                 Final Evidence
                        |
                        v
              Local Llama 3.2 3B
                        |
                        v
                 Grounded Answer
```

---

## 3. Retrieval Pipeline

### 3.1 Document ingestion

The project stores the source documents under:

```text
data/raw_docs/
```

The documents are processed and divided into smaller chunks before indexing.

The current corpus contains:

- 15 documents
- 95 chunks

The processed chunks are stored in:

```text
data/processed/chunks.json
```

### 3.2 Dense retrieval

Dense semantic embeddings are generated using:

```text
BAAI/bge-base-en-v1.5
```

The embeddings are indexed using FAISS.

Dense retrieval is useful when the user's wording differs from the exact wording in the document.

Example:

```text
User: outside projects
Document: moonlighting
```

### 3.3 BM25 lexical retrieval

BM25 provides lexical retrieval based on terms appearing in the indexed document metadata and text.

This complements semantic retrieval by giving stronger signals when important words directly occur in the source.

### 3.4 Reciprocal Rank Fusion

The dense and BM25 rankings are combined using Reciprocal Rank Fusion.

The final implementation uses:

```text
65% Dense retrieval
35% BM25 retrieval
```

This weighting was selected after evaluation showed that increasing the dense contribution improved retrieval coverage for difficult queries while retaining useful lexical matching.

### 3.5 BGE reranking

The system uses:

```text
BAAI/bge-reranker-base
```

The reranker evaluates query-passage relevance.

It is intentionally retained as an **evidence signal** rather than completely replacing the primary RRF retrieval ranking.

---

## 4. Corrective Retrieval / CRAG

The pipeline performs an initial retrieval pass and evaluates the retrieved evidence.

When the evidence is weak, the system activates corrective retrieval.

```text
Initial question
       |
       v
Initial retrieval
       |
       v
Evidence evaluation
       |
   Weak evidence?
       |
      Yes
       v
Query rewriting
       |
       v
Terminology expansion
       |
       v
Corrective retrieval
```

This helps with queries where the user's wording does not match the terminology used in the source documents.

### Example

Question:

```text
Can employees work on outside projects?
```

The source policy is organized under:

```text
moonlighting.md
```

The system uses controlled terminology expansion to connect the user's phrase **outside projects** with the policy terminology **moonlighting**.

The result is successful retrieval of the relevant `moonlighting.md` sections.

---

## 5. Query Rewriting

A locally hosted Llama model is also used to rewrite difficult questions into concise retrieval-oriented queries.

The rewrite stage is designed to:

- Preserve the user's intent.
- Preserve important policy intent such as allowed, prohibited, required, eligibility, and consequences.
- Identify domain-specific terminology.
- Avoid answering the question.
- Produce one search query.

The project also uses a small controlled terminology map for known document vocabulary.

This provides deterministic query expansion for important domain-specific mappings rather than relying entirely on the small local LLM to infer terminology.

---

## 6. Local LLM Generation

The project uses:

```text
Llama 3.2 3B Instruct
Q4_K_M GGUF
```

The model is loaded locally using `llama.cpp` through `llama-cpp-python`.

Generation is constrained by a prompt that instructs the model to:

- Use only the supplied context.
- Avoid outside knowledge.
- Avoid invented facts.
- State when information is not available.
- Keep answers concise.
- Reference supporting documents and sections when appropriate.

This makes the final response grounded in the retrieved evidence.

---

## 7. Evidence Evaluation

The project includes an evidence evaluation component that combines:

- BGE reranker relevance
- Keyword overlap with document metadata

The evaluator classifies the highest available evidence as:

```text
Strong
Moderate
Weak
```

Weak evidence can trigger corrective retrieval.

---

## 8. Handling Unsupported Questions

A key requirement is avoiding hallucinated enterprise-policy information.

For example:

```text
What is the company's annual revenue?
```

The evaluation corpus does not contain this information.

Instead of inventing a number, the system returns:

```text
The information was not found in the provided documents.
```

This behavior is important for policy and enterprise-document applications where unsupported answers can be misleading.

---

## 9. Streamlit Interface

The user interface is located at:

```text
ui/app.py
```

It provides:

- Question input
- Generated answer
- Evidence quality
- CRAG correction status
- Corrected search query when applicable
- Retrieved source documents
- Retrieved sections and supporting information

Run the interface from the project root with:

```bash
streamlit run ui/app.py
```

The application is intended to be run locally.

---

## 10. Evaluation

The project includes a 20-question evaluation set:

```text
evaluation/questions.json
```

The questions contain:

- 14 answerable questions with expected source documents
- 6 questions intentionally representing information that is unavailable or unsupported by the corpus

The final RRF evaluation produced:

```text
Questions evaluated : 20
Answerable questions: 14
Retrieval hits      : 14/14
Retrieval coverage  : 100%
Average latency     : 4.06 seconds
```

The 6 unsupported questions are not counted as retrieval hits because they intentionally have no expected source document.

### Final result

**14/14 answerable questions retrieved their expected document.**

This demonstrates full retrieval coverage across the answerable evaluation set.

---

## 11. Automated Tests

Automated validation is provided in:

```text
test_rag.py
```

The final test run produced:

```text
5 passed
```

The tests cover key behaviors including:

- Retriever result generation
- Medical-insurance retrieval
- Outside-project/moonlighting retrieval
- Unsupported-question handling
- Query rewriting

The older `test_*.py` scripts in the repository are also useful as manual component demonstrations.

---

## 12. Project Structure

```text
rag-decision-intelligence/
│
├── data/
│   ├── raw_docs/
│   └── processed/
│       └── chunks.json
│
├── embeddings/
│   ├── build_index.py
│   ├── index.faiss
│   └── metadata.json
│
├── evaluation/
│   ├── baseline_results.json
│   ├── questions.json
│   ├── rrf_results.json
│   ├── run_baseline.py
│   └── run_rrf.py
│
├── models/
│   └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
│
├── rag/
│   ├── chunker.py
│   ├── evidence.py
│   ├── generator.py
│   ├── loader.py
│   ├── pipeline.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── retriever_score_fusion.py
│   └── __init__.py
│
├── ui/
│   └── app.py
│
├── test_crag.py
├── test_evidence.py
├── test_generator.py
├── test_pipeline.py
├── test_rag.py
├── test_retrieval.py
├── test_rewrite.py
│
└── requirements.txt
```

---

## 13. Running the Project

### Create and activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the automated tests

```bash
python -m pytest test_rag.py -q
```

Expected result:

```text
5 passed
```

### Run the Streamlit application

```bash
streamlit run ui/app.py
```

### Run the final evaluation

```bash
python evaluation/run_rrf.py
```

---

## 14. Example Questions

### Answerable

```text
What type of medical insurance is provided to employees?
```

Expected source:

```text
benefits-and-perks.md
```

### Corrective retrieval

```text
Can employees work on outside projects?
```

Expected source:

```text
moonlighting.md
```

This query demonstrates corrective retrieval and terminology expansion.

### Unsupported

```text
What is the company's annual revenue?
```

Expected behavior:

```text
The information was not found in the provided documents.
```

---

## 15. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Dense embeddings | BAAI/bge-base-en-v1.5 |
| Vector search | FAISS |
| Lexical search | BM25 |
| Rank fusion | Reciprocal Rank Fusion |
| Reranking | BAAI/bge-reranker-base |
| Generation | Llama 3.2 3B Instruct |
| Local inference | llama.cpp / llama-cpp-python |
| UI | Streamlit |
| Testing | pytest |
| Data format | Markdown / JSON |

---

## 16. Limitations

The current system has several practical limitations:

1. The local 3B LLM is relatively small and may produce weaker query rewrites than larger instruction-tuned models.
2. The terminology expansion map is currently controlled and project-specific rather than automatically learned.
3. Retrieval latency includes model loading/inference overhead and may vary between runs.
4. The evaluation corpus is relatively small at 15 documents and 95 chunks.
5. The current evaluation primarily measures expected-document retrieval coverage rather than a large-scale human judgment of answer quality.
6. Production deployment would require stronger observability, security controls, model-serving infrastructure, and larger evaluation datasets.

---

## 17. Future Improvements

Potential future improvements include:

- Expand the evaluation dataset.
- Add automated answer-faithfulness and citation evaluation.
- Learn terminology mappings from the document corpus.
- Add metadata filtering.
- Improve chunking for long policy documents.
- Add conversation history when multi-turn questions are required.
- Add document versioning and freshness tracking.
- Use a larger local or hosted instruction model when hardware permits.
- Add latency and retrieval-quality monitoring.
- Add automated regression testing for retrieval quality.

---

## 18. Project Summary

> Built a local enterprise-policy RAG system combining FAISS dense retrieval, BM25 lexical search, 65/35 Reciprocal Rank Fusion, BGE reranking, evidence evaluation, CRAG-style corrective retrieval, query rewriting, terminology expansion, and a locally hosted Llama 3.2 3B model; evaluated on 20 questions with 14/14 retrieval coverage for answerable queries and 4.06s average latency.

---

## 19. Key Takeaway

The project demonstrates a complete local RAG workflow rather than a basic LLM chatbot:

```text
Retrieve
   ↓
Fuse
   ↓
Rerank
   ↓
Evaluate evidence
   ↓
Correct weak retrieval
   ↓
Generate from evidence
   ↓
Refuse unsupported information
```

The final system is designed to prioritize **retrieval quality, evidence grounding, and controlled generation** for enterprise-document question answering.
