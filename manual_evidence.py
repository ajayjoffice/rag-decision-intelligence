from rag.retriever import Retriever
from rag.evidence import EvidenceEvaluator


def run_evidence_test():
    retriever = Retriever()
    evaluator = EvidenceEvaluator()

    question = input("\nEnter your question: ").strip()

    results = retriever.search(
        question,
        top_k=5
    )

    evaluation = evaluator.evaluate(
        results,
        question
    )

    print("\n" + "=" * 80)
    print("EVIDENCE EVALUATION")
    print("=" * 80)

    print(f"Quality : {evaluation['quality']}")
    print(f"Score   : {evaluation['score']:.4f}")
    print(f"Reason  : {evaluation['reason']}")

    print("\nTOP RESULTS")

    for i, result in enumerate(results, start=1):
        print(
            f"\n[{i}] "
            f"{result['document_name']} | "
            f"{result['section']}"
        )

        print(
            f"Reranker: "
            f"{result['reranker_score']:.4f}"
        )


if __name__ == "__main__":
    run_evidence_test()