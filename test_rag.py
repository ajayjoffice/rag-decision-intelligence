from rag.retriever import Retriever
from rag.generator import LocalLLM
from rag.pipeline import RAGPipeline


def test_retriever_returns_results():
    retriever = Retriever()

    results = retriever.search(
        "What type of medical insurance is provided to employees?",
        top_k=5
    )

    assert len(results) > 0


def test_medical_insurance_retrieval():
    retriever = Retriever()

    results = retriever.search(
        "What type of medical insurance is provided to employees?",
        top_k=5
    )

    documents = [
        result["document_name"]
        for result in results
    ]

    assert "benefits-and-perks.md" in documents


def test_outside_projects_retrieval():
    pipeline = RAGPipeline()

    result = pipeline.answer(
        "Can employees work on outside projects?",
        top_k=5
    )

    documents = [
        source["document_name"]
        for source in result["sources"]
    ]

    assert "moonlighting.md" in documents


def test_unanswerable_question():
    pipeline = RAGPipeline()

    result = pipeline.answer(
        "What is the company's annual revenue?",
        top_k=5
    )

    answer = result["answer"].lower()

    assert (
        "not found" in answer
        or "information was not found" in answer
    )


def test_query_rewrite():
    llm = LocalLLM()

    rewritten = llm.rewrite_query(
        "Can employees work on outside projects?"
    )

    assert isinstance(rewritten, str)
    assert len(rewritten.strip()) > 0
