from rag.pipeline import RAGPipeline


pipeline = RAGPipeline()


questions = [
    "What type of medical insurance is provided to employees?",
    "How are employee promotions handled?",
    "What is the company's annual revenue?"
]


for question in questions:

    result = pipeline.answer(
        question,
        top_k=5
    )

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\nINITIAL EVIDENCE")
    print(
        f"Quality: "
        f"{result['initial_evidence']['quality']}"
    )

    print(
        f"Score: "
        f"{result['initial_evidence']['score']:.4f}"
    )

    print("\nCORRECTION USED")
    print(result["correction_used"])

    if result["rewritten_query"]:
        print(
            f"Rewritten query: "
            f"{result['rewritten_query']}"
        )

    print("\nFINAL EVIDENCE")
    print(
        f"Quality: "
        f"{result['evidence']['quality']}"
    )

    print(
        f"Score: "
        f"{result['evidence']['score']:.4f}"
    )

    print("\nFINAL ANSWER")
    print(result["answer"])

    print("\nSOURCES")

    for i, source in enumerate(
        result["sources"],
        start=1
    ):

        print(
            f"[{i}] "
            f"{source['document_name']} | "
            f"{source['section']} | "
            f"Reranker: "
            f"{source['reranker_score']:.4f}"
        )
