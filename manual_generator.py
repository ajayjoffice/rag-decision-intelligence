from rag.generator import LocalLLM

llm = LocalLLM()

chunks = [
    {
        "document_name": "moonlighting.md",
        "section": "OK",
        "text": (
            "Occasional side gigs, whether free or paid, are fine. "
            "Speaking gigs and other outside projects are also acceptable."
        ),
    }
]

question = "Can employees work on outside projects?"

answer = llm.generate(question, chunks)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(answer)
