import json
import time
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline


QUESTIONS_FILE = Path("evaluation/questions.json")
OUTPUT_FILE = Path("evaluation/rrf_results.json")


def main():

    # --------------------------------
    # Load evaluation questions
    # --------------------------------

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        questions = json.load(file)

    # --------------------------------
    # Initialize RAG once
    # --------------------------------

    pipeline = RAGPipeline()

    results = []

    print("\n" + "=" * 80)
    print("RRF RAG EVALUATION")
    print("=" * 80)

    # --------------------------------
    # Run every evaluation question
    # --------------------------------

    for item in questions:

        question = item["question"]

        print(
            f"\n[{item['id']}/"
            f"{len(questions)}] "
            f"{question}"
        )

        start_time = time.perf_counter()

        result = pipeline.answer(
            question,
            top_k=5
        )

        latency = time.perf_counter() - start_time

        sources = result["sources"]

        # --------------------------------
        # Retrieval information
        # --------------------------------

        retrieved_documents = [
            source["document_name"]
            for source in sources
        ]

        retrieved_sections = [
            source["section"]
            for source in sources
        ]

        final_scores = [
            source.get(
                "final_score",
                0.0
            )
            for source in sources
        ]

        reranker_scores = [
            source.get(
                "reranker_score",
                0.0
            )
            for source in sources
        ]

        max_final_score = (
            max(final_scores)
            if final_scores
            else None
        )

        max_reranker_score = (
            max(reranker_scores)
            if reranker_scores
            else None
        )

        # --------------------------------
        # Check expected document retrieval
        # --------------------------------

        expected_documents = set(
            item["expected_documents"]
        )

        retrieved_expected = bool(
            expected_documents.intersection(
                retrieved_documents
            )
        )

        # --------------------------------
        # Store result
        # --------------------------------

        evaluation_result = {
            "id": item["id"],
            "question": question,
            "category": item["category"],
            "expected_documents": item[
                "expected_documents"
            ],
            "retrieved_documents": retrieved_documents,
            "retrieved_sections": retrieved_sections,
            "retrieved_expected_document": (
                retrieved_expected
            ),
            "max_final_score": (
                max_final_score
            ),
            "max_reranker_score": (
                max_reranker_score
            ),
            "correction_used": result.get(
                "correction_used",
                False
            ),
            "rewritten_query": result.get(
                "rewritten_query"
            ),
            "answer": result["answer"],
            "latency_seconds": round(
                latency,
                3
            )
        }

        results.append(
            evaluation_result
        )

        print(
            f"Expected document retrieved: "
            f"{retrieved_expected}"
        )

        print(
            f"Max RRF score: "
            f"{max_final_score}"
        )

        print(
            f"Max reranker score: "
            f"{max_reranker_score}"
        )

        print(
            f"CRAG correction used: "
            f"{result.get('correction_used', False)}"
        )

        print(
            f"Latency: "
            f"{latency:.2f}s"
        )

    # --------------------------------
    # Save results
    # --------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------
    # Summary
    # --------------------------------

    retrieval_hits = sum(
        result["retrieved_expected_document"]
        for result in results
        if result["expected_documents"]
    )

    retrieval_total = sum(
        bool(result["expected_documents"])
        for result in results
    )

    average_latency = (
        sum(
            result["latency_seconds"]
            for result in results
        )
        / len(results)
    )

    print("\n" + "=" * 80)
    print("RRF SUMMARY")
    print("=" * 80)

    print(
        f"Questions evaluated : "
        f"{len(results)}"
    )

    print(
        f"Retrieval hits      : "
        f"{retrieval_hits}/{retrieval_total}"
    )

    print(
        f"Average latency     : "
        f"{average_latency:.2f}s"
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()

