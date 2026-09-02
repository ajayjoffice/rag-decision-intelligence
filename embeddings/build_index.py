import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_FILE = Path("embeddings/index.faiss")
METADATA_FILE = Path("embeddings/metadata.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def build_index():

    print("=" * 60)
    print("SECTION-AWARE BGE EMBEDDINGS + FAISS INDEX")
    print("=" * 60)

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    # Include document and section context in the embedding text.
    texts = []

    for chunk in chunks:
        retrieval_text = (
            f"Document: {chunk['document_name']}\n"
            f"Section: {chunk['section']}\n\n"
            f"{chunk['text']}"
        )

        texts.append(retrieval_text)

    print("\nExample retrieval representation:")
    print("-" * 60)
    print(texts[0][:500])
    print("-" * 60)

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    print("\nGenerating section-aware document embeddings...")

    embeddings = model.encode_document(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    print(
        f"Embedding matrix shape: {embeddings.shape}"
    )

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors
    # = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors stored: {index.ntotal}"
    )

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)

    print(f"Index: {INDEX_FILE}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Vector dimension: {dimension}")
    print(f"Total vectors: {index.ntotal}")


if __name__ == "__main__":
    build_index()