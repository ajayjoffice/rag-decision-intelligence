import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

INDEX_PATH = Path("embeddings/index.faiss")
METADATA_PATH = Path("embeddings/metadata.json")


class Retriever:

    def __init__(self):
        print("Loading retrieval system...")

        # -----------------------------------------------------
        # Dense embedding model
        # -----------------------------------------------------

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        # -----------------------------------------------------
        # FAISS index
        # -----------------------------------------------------

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        # -----------------------------------------------------
        # Chunk metadata
        # -----------------------------------------------------

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            self.metadata = json.load(f)

        # -----------------------------------------------------
        # Cross-encoder reranker
        # -----------------------------------------------------

        print(
            f"Loading reranker: {RERANKER_MODEL}"
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

        print("Reranker loaded.")

        # -----------------------------------------------------
        # BM25 index
        # -----------------------------------------------------

        self.bm25_documents = [
            self._tokenize(
                f"{chunk['document_name']} "
                f"{chunk['section']} "
                f"{chunk['text']}"
            )
            for chunk in self.metadata
        ]

        self.bm25 = BM25Okapi(
            self.bm25_documents
        )

        print(
            f"Loaded {self.index.ntotal} vectors."
        )

        print(
            f"BM25 documents indexed: "
            f"{len(self.metadata)}"
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================

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

    # =========================================================
    # DENSE RETRIEVAL
    # =========================================================

    def dense_search(
        self,
        query: str,
        top_k: int = 15
    ):

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            np.asarray(
                query_embedding,
                dtype="float32"
            ),
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            result = self.metadata[index].copy()

            result["similarity_score"] = float(
                score
            )

            result["dense_rank"] = (
                len(results) + 1
            )

            results.append(result)

        return results

    # =========================================================
    # BM25 / LEXICAL RETRIEVAL
    # =========================================================

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

        ranked_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in ranked_indices:

            result = self.metadata[index].copy()

            result["bm25_score"] = float(
                scores[index]
            )

            result["bm25_rank"] = (
                len(results) + 1
            )

            results.append(result)

        return results

    # =========================================================
    # HYBRID CANDIDATE GENERATION
    # =========================================================

    def hybrid_search(
        self,
        query: str,
        top_k: int = 15
    ):

        dense_results = self.dense_search(
            query,
            top_k=top_k
        )

        lexical_results = self.lexical_search(
            query,
            top_k=top_k
        )

        candidates = {}

        # -----------------------------------------------------
        # Add dense candidates
        # -----------------------------------------------------

        for result in dense_results:

            chunk_id = result["chunk_id"]

            candidates[chunk_id] = result.copy()

            candidates[chunk_id][
                "retrieval_method"
            ] = "dense"

        # -----------------------------------------------------
        # Merge BM25 candidates
        # -----------------------------------------------------

        for result in lexical_results:

            chunk_id = result["chunk_id"]

            if chunk_id not in candidates:

                candidates[chunk_id] = result.copy()

                candidates[chunk_id][
                    "retrieval_method"
                ] = "bm25"

            else:

                candidates[chunk_id][
                    "bm25_score"
                ] = result[
                    "bm25_score"
                ]

                candidates[chunk_id][
                    "bm25_rank"
                ] = result[
                    "bm25_rank"
                ]

                candidates[chunk_id][
                    "retrieval_method"
                ] = "hybrid"

        return list(
            candidates.values()
        )

    # =========================================================
    # CROSS-ENCODER RERANKING
    # =========================================================

    def rerank(
        self,
        query: str,
        results: list
    ):

        if not results:
            return []

        pairs = []

        for result in results:

            passage = (
                f"Document: "
                f"{result['document_name']}\n"
                f"Section: "
                f"{result['section']}\n"
                f"{result['text']}"
            )

            pairs.append(
                [
                    query,
                    passage
                ]
            )

        scores = self.reranker.predict(
            pairs
        )

        reranked = []

        for result, score in zip(
            results,
            scores
        ):

            result = result.copy()

            result["reranker_score"] = float(
                score
            )

            reranked.append(result)

        reranked.sort(
            key=lambda x: x[
                "reranker_score"
            ],
            reverse=True
        )

        for rank, result in enumerate(
            reranked,
            start=1
        ):

            result["reranker_rank"] = rank

        return reranked

    # =========================================================
    # RECIPROCAL RANK FUSION
    #
    # Final retrieval ranking uses ONLY:
    #
    #   50% Dense rank
    #   50% BM25 rank
    #
    # The reranker is intentionally NOT used here because
    # testing showed that it incorrectly demotes the
    # "Pay & Promotions" evidence.
    # =========================================================

    def _rrf_score(
        self,
        dense_rank,
        bm25_rank,
        k=60
    ):

        score = 0.0

        if dense_rank is not None:

            score += (
                0.65 /
                (k + dense_rank)
            )

        if bm25_rank is not None:

            score += (
                0.35 /
                (k + bm25_rank)
            )

        return score

    # =========================================================
    # FINAL SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        # -----------------------------------------------------
        # Step 1: Generate hybrid candidate pool
        # -----------------------------------------------------

        candidates = self.hybrid_search(
            query,
            top_k=15
        )

        # -----------------------------------------------------
        # Step 2: Calculate reranker scores
        #
        # Important:
        # Reranker is retained as an evidence signal.
        # It does NOT control final retrieval ranking.
        # -----------------------------------------------------

        reranked = self.rerank(
            query,
            candidates
        )

        # -----------------------------------------------------
        # Step 3: Build rank lookup
        # -----------------------------------------------------

        dense_ranks = {
            result["chunk_id"]:
            result.get("dense_rank")
            for result in candidates
        }

        bm25_ranks = {
            result["chunk_id"]:
            result.get("bm25_rank")
            for result in candidates
        }

        reranker_scores = {
            result["chunk_id"]:
            result.get(
                "reranker_score",
                0.0
            )
            for result in reranked
        }

        reranker_ranks = {
            result["chunk_id"]:
            result.get("reranker_rank")
            for result in reranked
        }

        # -----------------------------------------------------
        # Step 4: Calculate final RRF score
        # -----------------------------------------------------

        final_results = []

        for result in candidates:

            chunk_id = result["chunk_id"]

            dense_rank = dense_ranks.get(
                chunk_id
            )

            bm25_rank = bm25_ranks.get(
                chunk_id
            )

            final_score = self._rrf_score(
                dense_rank,
                bm25_rank
            )

            enriched = result.copy()

            enriched["reranker_score"] = (
                reranker_scores.get(
                    chunk_id,
                    0.0
                )
            )

            enriched["reranker_rank"] = (
                reranker_ranks.get(
                    chunk_id
                )
            )

            enriched["final_score"] = float(
                final_score
            )

            enriched["dense_rank"] = (
                dense_rank
            )

            enriched["bm25_rank"] = (
                bm25_rank
            )

            final_results.append(
                enriched
            )

        # -----------------------------------------------------
        # Step 5: Sort by final RRF score
        # -----------------------------------------------------

        final_results.sort(
            key=lambda x: x[
                "final_score"
            ],
            reverse=True
        )

        # -----------------------------------------------------
        # Step 6: Return Top-K
        # -----------------------------------------------------

        return final_results[:top_k]


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    retriever = Retriever()

    query = (
        "How are employee promotions handled?"
    )

    results = retriever.search(
        query,
        top_k=5
    )

    print(
        "\nQuery:",
        query
    )

    print(
        "\nTop Results:\n"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['document_name']} | "
            f"{result['section']}"
        )

        print(
            f"   Final RRF: "
            f"{result['final_score']:.6f}"
        )

        print(
            f"   Dense rank: "
            f"{result.get('dense_rank')}"
        )

        print(
            f"   BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"   Reranker rank: "
            f"{result.get('reranker_rank')}"
        )

        print(
            f"   Reranker score: "
            f"{result.get('reranker_score', 0):.6f}"
        )

        print()