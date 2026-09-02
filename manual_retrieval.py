from rag.retriever import Retriever

r = Retriever()

queries = [
    "What type of medical insurance is provided to employees?",
    "How are employee promotions handled?",
    "What is the companys annual revenue?",
]

for q in queries:
    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("=" * 80)

    results = r.search(q, top_k=5)

    for i, x in enumerate(results, start=1):
        print(
            f"{i}. {x['document_name']} | "
            f"{x['section']} | "
            f"Final: {x['final_score']:.6f} | "
            f"Dense Rank: {x.get('dense_rank')} | "
            f"BM25 Rank: {x.get('bm25_rank')} | "
            f"Reranker Rank: {x.get('reranker_rank')} | "
            f"Reranker: {x.get('reranker_score', 0):.6f}"
        )
