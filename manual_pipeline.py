from rag.pipeline import RAGPipeline


def run_pipeline_test():
    pipeline = RAGPipeline()

    question = input(
        "\nEnter your question: "
    ).strip()

    result = pipeline.answer(
        question,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n" + "=" * 80)
    print("SOURCES")
    print("=" * 80)

    for i, source in enumerate(
        result["sources"],
        start=1
    ):

        print(f"\n[{i}]")
        print(f"Document : {source['document_name']}")
        print(f"Section  : {source['section']}")
        print(
            f"FAISS    : "
            f"{source['similarity_score']:.4f}"
        )
        print(
            f"Reranker : "
            f"{source['reranker_score']:.4f}"
        )


if __name__ == "__main__":
    run_pipeline_test()