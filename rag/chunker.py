import json
import re
from pathlib import Path

from rag.loader import load_markdown_documents


OUTPUT_FILE = Path("data/processed/chunks.json")

MAX_WORDS = 350
OVERLAP_WORDS = 50


def clean_markdown(text):
    """
    Basic Markdown cleanup while preserving meaningful text.
    """

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove Markdown links but keep visible text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_sections(text):
    """
    Split Markdown document based on headings.

    Returns:
        list of (section_title, section_text)
    """

    lines = text.splitlines()

    sections = []

    current_heading = "Introduction"
    current_content = []

    for line in lines:

        heading_match = re.match(
            r"^(#{1,6})\s+(.+)$",
            line.strip()
        )

        if heading_match:

            if current_content:
                sections.append(
                    (
                        current_heading,
                        "\n".join(current_content).strip()
                    )
                )

            current_heading = heading_match.group(2).strip()
            current_content = []

        else:
            current_content.append(line)

    if current_content:
        sections.append(
            (
                current_heading,
                "\n".join(current_content).strip()
            )
        )

    return sections


def word_chunks(text, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + max_words, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_chunks():

    documents = load_markdown_documents()

    all_chunks = []

    chunk_id = 0

    for document in documents:

        cleaned_text = clean_markdown(
            document["text"]
        )

        sections = split_into_sections(
            cleaned_text
        )

        for section_title, section_text in sections:

            if not section_text.strip():
                continue

            chunks = word_chunks(section_text)

            for chunk in chunks:

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_name": document[
                            "document_name"
                        ],
                        "document_path": document[
                            "document_path"
                        ],
                        "section": section_title,
                        "text": chunk,
                    }
                )

                chunk_id += 1

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_chunks()