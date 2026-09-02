from rag.generator import LocalLLM


llm = LocalLLM()

questions = [
    "How are employee promotions handled?",
    "What does the company say about working independently?",
    "Can employees work on outside projects?"
]

for question in questions:

    rewritten = llm.rewrite_query(
        question
    )

    print("\n" + "=" * 80)
    print("ORIGINAL:")
    print(question)

    print("\nREWRITTEN:")
    print(rewritten)
