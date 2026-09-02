from rag.retriever import Retriever
from rag.generator import LocalLLM
from rag.evidence import EvidenceEvaluator


class RAGPipeline:

    def __init__(self):

        print("\nInitializing RAG pipeline...")

        # ==========================================
        # COMPONENT 1 — RETRIEVAL
        # ==========================================

        # Dense embeddings -> FAISS
        # BM25 lexical retrieval
        # 65/35 Reciprocal Rank Fusion
        # BGE reranker retained as an evidence signal
        self.retriever = Retriever()

        # ==========================================
        # COMPONENT 2 — LOCAL LLM
        # ==========================================

        # Shared local Llama model used for:
        # - query rewriting
        # - final answer generation
        self.llm = LocalLLM()

        # ==========================================
        # COMPONENT 3 — EVIDENCE EVALUATION
        # ==========================================

        self.evidence = EvidenceEvaluator()

        # Controlled terminology mappings for
        # domain-specific vocabulary used in the
        # source documents.
        self.terminology_map = {
            "outside projects": "moonlighting",
            "outside project": "moonlighting",
            "outside work": "moonlighting",
            "side gigs": "moonlighting",
            "side gig": "moonlighting",
        }

        print("RAG pipeline ready.")

    def _expand_query(self, query: str):
        """
        Add known document terminology when a user
        uses an equivalent phrase.
        """

        expanded_query = query.lower()

        for phrase, terminology in self.terminology_map.items():

            if phrase in expanded_query:
                if terminology not in expanded_query:
                    expanded_query = (
                        f"{query} {terminology}"
                    )

                break

        return expanded_query

    def answer(
        self,
        question: str,
        top_k: int = 5
    ):

        # ==========================================
        # STAGE 1 — INITIAL RETRIEVAL
        # ==========================================

        initial_results = self.retriever.search(
            question,
            top_k=max(top_k, 15)
        )

        initial_evidence = self.evidence.evaluate(
            initial_results,
            question
        )

        # ==========================================
        # STAGE 2 — CORRECTIVE RETRIEVAL / CRAG
        # ==========================================

        final_results = initial_results
        final_evidence = initial_evidence

        rewritten_query = None
        correction_used = False

        if initial_evidence["quality"] == "weak":

            correction_used = True

            # First let the local LLM rewrite the query.
            rewritten_query = self.llm.rewrite_query(
                question
            )

            rewritten_query = (
                rewritten_query
                .replace('"', "")
                .replace("'", "")
                .replace("\n", " ")
                .strip()
            )

            # Add controlled domain terminology.
            corrected_query = self._expand_query(
                rewritten_query
            )

            corrected_results = self.retriever.search(
                corrected_query,
                top_k=max(top_k, 15)
            )

            corrected_evidence = self.evidence.evaluate(
                corrected_results,
                question
            )

            # Keep corrected retrieval when it provides
            # stronger evidence.
            if (
                corrected_evidence["score"]
                > initial_evidence["score"]
            ):
                final_results = corrected_results
                final_evidence = corrected_evidence

        # ==========================================
        # STAGE 3 — FINAL EVIDENCE SELECTION
        # ==========================================

        final_results = final_results[:top_k]

        final_evidence = self.evidence.evaluate(
            final_results,
            question
        )

        # ==========================================
        # STAGE 4 — GENERATION
        # ==========================================

        answer = self.llm.generate(
            question,
            final_results
        )

        # ==========================================
        # STAGE 5 — STRUCTURED OUTPUT
        # ==========================================

        return {
            "question": question,
            "answer": answer,
            "sources": final_results,
            "evidence": final_evidence,
            "initial_evidence": initial_evidence,
            "rewritten_query": rewritten_query,
            "correction_used": correction_used
        }