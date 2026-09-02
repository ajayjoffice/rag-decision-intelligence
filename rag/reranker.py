from sentence_transformers import CrossEncoder


MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:

    def __init__(self):

        print(
            f"Loading reranker: {MODEL_NAME}"
        )

        self.model = CrossEncoder(
            MODEL_NAME
        )

        print("Reranker loaded.")

    def rerank(
        self,
        query: str,
        results: list,
        top_k: int = 5
    ):

        if not results:
            return []

        # Include metadata so the reranker understands
        # the semantic meaning of the section/document.
        pairs = [
            [
                query,
                (
                    f"Document: {result['document_name']}\n"
                    f"Section: {result['section']}\n"
                    f"{result['text']}"
                )
            ]
            for result in results
        ]

        scores = self.model.predict(
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
            key=lambda x: x["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]