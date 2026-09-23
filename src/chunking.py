# src/chunking.py

from pathlib import Path
from typing import List, Dict, Any
import json
import re


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

INPUT_FILE = PROCESSED_DATA_DIR / "all_pages.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "all_chunks.jsonl"


# ---------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------

# Approximation:
# 1 token ~= 0.75 words
# So 500-800 tokens is roughly 375-600 words.
#
# We use 450 words per chunk with 80 words overlap.
CHUNK_SIZE_WORDS = 450
CHUNK_OVERLAP_WORDS = 80


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving readable structure.
    """

    if not text:
        return ""

    # Replace repeated whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


def split_into_words(text: str) -> List[str]:
    """
    Split text into words.
    """
    return text.split()


# ---------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS
) -> List[str]:
    """
    Split text into overlapping word chunks.

    Example:
        chunk_size = 450 words
        overlap = 80 words

    Returns:
        List[str]
    """

    words = split_into_words(text)

    if not words:
        return []

    # If text is already small enough
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        # Stop if we've reached the end
        if end >= len(words):
            break

        # Move forward while keeping overlap
        start = end - overlap

    return chunks


# ---------------------------------------------------------
# Read page-level dataset
# ---------------------------------------------------------

def load_pages(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load page-level records from JSONL.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Run pdf_processor.py first."
        )

    records = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                records.append(record)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

    return records


# ---------------------------------------------------------
# Convert pages into chunks
# ---------------------------------------------------------

def create_chunks(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert page-level records into chunk-level records.

    Preserves:
        - company
        - ticker
        - document
        - file_name
        - page

    Adds:
        - chunk_id
        - chunk_index
        - text
    """

    all_chunks = []

    global_chunk_id = 0

    for page_record in pages:

        raw_text = page_record.get("text", "")
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue

        page_chunks = chunk_text(cleaned_text)

        for chunk_index, chunk in enumerate(page_chunks):

            global_chunk_id += 1

            chunk_record = {
                "chunk_id": global_chunk_id,
                "company": page_record.get("company"),
                "ticker": page_record.get("ticker"),
                "document": page_record.get("document"),
                "file_name": page_record.get("file_name"),
                "page": page_record.get("page"),
                "chunk_index": chunk_index,
                "word_count": len(chunk.split()),
                "text": chunk,
            }

            all_chunks.append(chunk_record)

    return all_chunks


# ---------------------------------------------------------
# Save chunks
# ---------------------------------------------------------

def save_jsonl(
    records: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save chunk records as JSONL.
    """

    with open(output_path, "w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(f"Saved chunks to: {output_path}")


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def print_statistics(
    pages: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]]
) -> None:
    """
    Print useful chunking statistics.
    """

    print("\n" + "=" * 60)
    print("Chunking Statistics")
    print("=" * 60)

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    if chunks:

        word_counts = [
            chunk["word_count"]
            for chunk in chunks
        ]

        average_words = sum(word_counts) / len(word_counts)

        print(
            f"Average words per chunk: "
            f"{average_words:.2f}"
        )

        print(
            f"Smallest chunk: "
            f"{min(word_counts)} words"
        )

        print(
            f"Largest chunk: "
            f"{max(word_counts)} words"
        )

    # Company-level stats
    company_counts = {}

    for chunk in chunks:
        ticker = chunk.get("ticker", "UNKNOWN")

        company_counts[ticker] = (
            company_counts.get(ticker, 0) + 1
        )

    print("\nChunks per company:")

    for ticker, count in company_counts.items():
        print(f"  {ticker}: {count}")

    print("=" * 60)


# ---------------------------------------------------------
# Preview
# ---------------------------------------------------------

def preview_chunks(
    chunks: List[Dict[str, Any]],
    number: int = 3
) -> None:
    """
    Print a few example chunks.
    """

    print("\n" + "=" * 60)
    print("Chunk Preview")
    print("=" * 60)

    for chunk in chunks[:number]:

        print(
            f"\nChunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Company: {chunk['ticker']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            f"Words: {chunk['word_count']}"
        )

        print("\nText:")

        preview_text = chunk["text"][:500]

        print(preview_text)

        if len(chunk["text"]) > 500:
            print("...")

        print("-" * 60)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("StockLens Chunking Pipeline")
    print("=" * 60)

    print(f"\nInput: {INPUT_FILE}")

    pages = load_pages(INPUT_FILE)

    print(
        f"Loaded {len(pages)} page records."
    )

    chunks = create_chunks(pages)

    save_jsonl(
        records=chunks,
        output_path=OUTPUT_FILE
    )

    print_statistics(
        pages=pages,
        chunks=chunks
    )

    preview_chunks(
        chunks=chunks,
        number=3
    )

    print("\nChunking complete.")


if __name__ == "__main__":
    main()