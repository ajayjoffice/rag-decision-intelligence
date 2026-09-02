import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.reranker import Reranker


INDEX_FILE = Path("embeddings/index.faiss")
METADATA_FILE = Path("embeddings/metadata.json")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class Retriever:

    def __init__(self):

        print("Loading retrieval system...")

        # -----------------------------
        # Embedding model
        # -----------------------------

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # -----------------------------
        # FAISS index
        # -----------------------------

        self.index = faiss.read_index(
            str(INDEX_FILE)
        )

        # -----------------------------
        # Metadata
        # -----------------------------

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(file)

        # -----------------------------
        # BM25 lexical index
        # -----------------------------

        self.bm25_documents = []

        for chunk in self.metadata:

            retrieval_text = (
                f"{chunk['document_name']} "
                f"{chunk['section']} "
                f"{chunk['text']}"
            )

            self.bm25_documents.append(
                self._tokenize(retrieval_text)
            )

        self.bm25 = BM25Okapi(
            self.bm25_documents
        )

        # -----------------------------
        # Cross-encoder reranker
        # -----------------------------

        self.reranker = Reranker()

        print(
            f"Loaded {self.index.ntotal} vectors."
        )

        print(
            f"BM25 documents indexed: "
            f"{len(self.bm25_documents)}"
        )

    # =================================
    # Tokenization
    # =================================

    def _tokenize(self, text: str):

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "be",
            "can",
            "company",
            "do",
            "does",
            "employee",
            "employees",
            "for",
            "from",
            "how",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "work",
            "working",
        }

        tokens = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

        return [
            token
            for token in tokens
            if token not in stopwords
        ]

    # =================================
    # Dense retrieval
    # =================================

    def dense_search(
        self,
        query: str,
        top_k: int = 15
    ):

        query_embedding = (
            self.embedding_model.encode_query(
                query,
                normalize_embeddings=True
            )
        )

        query_embedding = np.asarray(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            result = self.metadata[index].copy()

            result["similarity_score"] = float(
                score
            )

            results.append(result)

        return results

    # =================================
    # BM25 retrieval
    # =================================

    def lexical_search(
        self,
        query: str,
        top_k: int = 15
    ):

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            result = self.metadata[index].copy()

            result["bm25_score"] = float(
                scores[index]
            )

            results.append(result)

        return results

    # =================================
    # Hybrid candidate retrieval
    # =================================

    def hybrid_search(
        self,
        query: str,
        dense_top_k: int = 15,
        lexical_top_k: int = 15
    ):

        dense_results = self.dense_search(
            query,
            top_k=dense_top_k
        )

        lexical_results = self.lexical_search(
            query,
            top_k=lexical_top_k
        )

        candidates = {}

        # Add dense candidates
        for result in dense_results:

            chunk_id = result["chunk_id"]

            candidates[chunk_id] = result.copy()

            candidates[chunk_id][
                "retrieval_method"
            ] = "dense"

        # Add BM25 candidates
        for result in lexical_results:

            chunk_id = result["chunk_id"]

            if chunk_id in candidates:

                candidates[chunk_id][
                    "bm25_score"
                ] = result["bm25_score"]

                candidates[chunk_id][
                    "retrieval_method"
                ] = "dense+bm25"

            else:

                candidates[chunk_id] = result.copy()

                candidates[chunk_id][
                    "retrieval_method"
                ] = "bm25"

        return list(
            candidates.values()
        )

    # =================================
    # Min-max normalization
    # =================================

    def _normalize_scores(
        self,
        results: list,
        field: str
    ):

        values = np.array(
            [
                float(result.get(field, 0.0))
                for result in results
            ],
            dtype="float32"
        )

        if len(values) == 0:
            return

        minimum = values.min()
        maximum = values.max()

        if maximum == minimum:

            normalized = np.ones(
                len(values),
                dtype="float32"
            )

        else:

            normalized = (
                (values - minimum)
                / (maximum - minimum)
            )

        for result, value in zip(
            results,
            normalized
        ):

            result[
                f"{field}_normalized"
            ] = float(value)

    # =================================
    # Score fusion
    # =================================

    def _fuse_scores(
        self,
        results: list
    ):

        self._normalize_scores(
            results,
            "similarity_score"
        )

        self._normalize_scores(
            results,
            "bm25_score"
        )

        self._normalize_scores(
            results,
            "reranker_score"
        )

        for result in results:

            dense_score = result.get(
                "similarity_score_normalized",
                0.0
            )

            bm25_score = result.get(
                "bm25_score_normalized",
                0.0
            )

            reranker_score = result.get(
                "reranker_score_normalized",
                0.0
            )

            # Initial engineering weights.
            # These will be calibrated using evaluation results.
            final_score = (
                0.35 * dense_score
                + 0.20 * bm25_score
                + 0.45 * reranker_score
            )

            result["final_score"] = float(
                final_score
            )

        results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return results

    # =================================
    # Final retrieval pipeline
    # =================================

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        # Stage 1:
        # Dense + lexical candidate generation
        candidates = self.hybrid_search(
            query,
            dense_top_k=15,
            lexical_top_k=15
        )

        # Stage 2:
        # Cross-encoder reranking
        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=len(candidates)
        )

        # Stage 3:
        # Normalize and fuse retrieval signals
        fused = self._fuse_scores(
            reranked
        )

        return fused[:top_k]


if __name__ == "__main__":

    retriever = Retriever()

    query = input(
        "\nEnter your question: "
    ).strip()

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n" + "=" * 75)
    print("HYBRID FUSED RETRIEVAL RESULTS")
    print("=" * 75)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(f"\n#{rank}")

        print(
            f"Final score: "
            f"{result.get('final_score', 0.0):.4f}"
        )

        print(
            f"FAISS similarity: "
            f"{result.get('similarity_score', 0.0):.4f}"
        )

        print(
            f"BM25 score: "
            f"{result.get('bm25_score', 0.0):.4f}"
        )

        print(
            f"Reranker score: "
            f"{result.get('reranker_score', 0.0):.4f}"
        )

        print(
            f"Retrieval method: "
            f"{result.get('retrieval_method', 'unknown')}"
        )

        print(
            f"Document: "
            f"{result['document_name']}"
        )

        print(
            f"Section: "
            f"{result['section']}"
        )

        print(
            f"Text: "
            f"{result['text'][:500]}"
        )