from pathlib import Path


RAW_DATA_DIR = Path("data/raw_docs")


def load_markdown_documents():
    """
    Load all Markdown documents from the raw data directory.

    Returns:
        list[dict]: Documents with filename, path and text.
    """

    documents = []

    for file_path in sorted(RAW_DATA_DIR.rglob("*.md")):

        try:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if not text:
                continue

            documents.append(
                {
                    "document_name": file_path.name,
                    "document_path": str(file_path),
                    "text": text,
                }
            )

        except Exception as error:
            print(f"Error reading {file_path}: {error}")

    return documents


if __name__ == "__main__":

    documents = load_markdown_documents()

    print("=" * 60)
    print("DOCUMENT LOADER")
    print("=" * 60)

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        word_count = len(document["text"].split())

        print(
            f"- {document['document_name']}: "
            f"{word_count:,} words"
        )