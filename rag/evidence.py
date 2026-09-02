import re


class EvidenceEvaluator:

    def _normalize(self, text: str):
        return set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9'-]*\b",
                text.lower()
            )
        )

    def _keyword_overlap(self, query: str, result: dict):
        query_terms = self._normalize(query)

        metadata = (
            f"{result.get('document_name', '')} "
            f"{result.get('section', '')}"
        )

        metadata_terms = self._normalize(metadata)

        if not query_terms:
            return 0.0

        return len(query_terms & metadata_terms) / len(query_terms)

    def score_result(self, query: str, result: dict):
        reranker_score = float(
            result.get("reranker_score", 0.0)
        )

        keyword_score = self._keyword_overlap(
            query,
            result
        )

        # Reranker remains the primary signal.
        # Metadata overlap provides a secondary signal.
        combined_score = (
            0.85 * reranker_score
            + 0.15 * keyword_score
        )

        scored = result.copy()

        scored["keyword_score"] = float(
            keyword_score
        )

        scored["evidence_score"] = float(
            combined_score
        )

        return scored

    def evaluate(self, results, query: str = ""):

        if not results:
            return {
                "quality": "weak",
                "score": 0.0,
                "reason": "No evidence retrieved."
            }

        scored_results = [
            self.score_result(query, result)
            for result in results
        ]

        max_score = max(
            result["evidence_score"]
            for result in scored_results
        )

        if max_score >= 0.20:
            quality = "strong"
        elif max_score >= 0.05:
            quality = "moderate"
        else:
            quality = "weak"

        return {
            "quality": quality,
            "score": float(max_score),
            "reason": (
                f"Highest evidence score: "
                f"{max_score:.4f}"
            ),
            "results": scored_results
        }