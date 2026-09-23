# src/bm25_retriever.py

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT
    / "processed_data"
    / "all_chunks.jsonl"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DEFAULT_TOP_K = 5

VALID_COMPANIES = {
    "CBA",
    "NAB",
    "BOTH",
}


# ---------------------------------------------------------
# Load chunks
# ---------------------------------------------------------

def load_chunks(
    file_path: Path
) -> List[Dict[str, Any]]:
    """
    Load chunk records from JSONL.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found:\n{file_path}\n\n"
            "Run pdf_processor.py and chunking.py first."
        )

    chunks = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}"
                ) from exc

            # Ignore extremely small chunks
            # because they usually have little
            # retrieval value.
            if record.get(
                "word_count",
                0
            ) < 20:
                continue

            chunks.append(record)

    return chunks


# ---------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------

def tokenize(
    text: str
) -> List[str]:
    """
    Simple tokenizer for BM25.

    - converts to lowercase
    - keeps words and numbers
    - removes punctuation
    """

    if not text:
        return []

    text = text.lower()

    tokens = re.findall(
        r"\b[a-z0-9.%$]+\b",
        text
    )

    return tokens


# ---------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------

class BM25Retriever:

    def __init__(
        self,
        chunks: List[Dict[str, Any]]
    ):
        """
        Build BM25 index from chunks.
        """

        self.chunks = chunks

        print(
            f"Building BM25 index for "
            f"{len(chunks)} chunks..."
        )

        self.tokenized_corpus = [
            tokenize(
                chunk["text"]
            )
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print(
            "BM25 index ready."
        )


    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search chunks using BM25.

        Args:
            query:
                User question.

            top_k:
                Number of results.

            company:
                CBA
                NAB
                BOTH / None

        Returns:
            Ranked list of matching chunks.
        """

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        results = []

        for index, score in enumerate(
            scores
        ):

            chunk = self.chunks[
                index
            ]

            ticker = (
                chunk
                .get(
                    "ticker",
                    ""
                )
                .upper()
            )

            # Company filtering
            if (
                company
                and company.upper()
                != "BOTH"
                and ticker
                != company.upper()
            ):
                continue

            result = {
                "score": float(
                    score
                ),
                "chunk_id": chunk.get(
                    "chunk_id"
                ),
                "company": chunk.get(
                    "company"
                ),
                "ticker": ticker,
                "document": chunk.get(
                    "document"
                ),
                "file_name": chunk.get(
                    "file_name"
                ),
                "page": chunk.get(
                    "page"
                ),
                "chunk_index": chunk.get(
                    "chunk_index"
                ),
                "text": chunk.get(
                    "text",
                    ""
                ),
            }

            results.append(
                result
            )

        # Sort highest score first
        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

def display_results(
    query: str,
    results: List[Dict[str, Any]]
) -> None:
    """
    Print BM25 search results clearly.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"QUESTION:\n{query}"
    )

    print(
        "=" * 80
    )

    if not results:

        print(
            "\nNo relevant results found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT #{rank}"
        )

        print(
            "-" * 80
        )

        print(
            f"Company: "
            f"{result['company']}"
        )

        print(
            f"Ticker: "
            f"{result['ticker']}"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document']}"
        )

        print(
            "\nRetrieved text:\n"
        )

        print(
            result["text"]
        )

        print(
            "\n"
            + "-" * 80
        )


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

def ask_company():

    while True:

        print("\nChoose company:")
        print("1. CBA")
        print("2. NAB")
        print("3. BOTH")
        print("4. EXIT")

        choice = input(
            "\nEnter 1, 2, 3, or 4: "
        ).strip().lower()

        if choice == "1":
            return "CBA"

        if choice == "2":
            return "NAB"

        if choice == "3":
            return "BOTH"

        if choice in {"4", "exit", "quit", "q"}:
            return "EXIT"

        print("Invalid choice. Please try again.")
# ---------------------------------------------------------
# Interactive search
# ---------------------------------------------------------

def interactive_search(
    retriever: BM25Retriever
) -> None:
    """
    Interactive command-line search.
    """

    print("\n" + "=" * 80)
    print("StockLens BM25 Retriever")
    print("=" * 80)

    print("\nType 'exit' to quit.")

    while True:

        company = ask_company()

        if company == "EXIT":
            print("\nExiting StockLens.")
            break

        print(f"\nSelected: {company}")

        query = input(
            "\nAsk a financial question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:
            print("\nExiting StockLens.")
            break

        if not query:
            print("Please enter a question.")
            continue

        results = retriever.search(
            query=query,
            top_k=DEFAULT_TOP_K,
            company=company
        )

        display_results(
            query=query,
            results=results
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(
        "=" * 80
    )

    print(
        "StockLens BM25 Retrieval System"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLoading chunks from:\n"
        f"{CHUNKS_FILE}"
    )

    chunks = load_chunks(
        CHUNKS_FILE
    )

    print(
        f"\nLoaded "
        f"{len(chunks)} usable chunks."
    )

    retriever = BM25Retriever(
        chunks
    )

    interactive_search(
        retriever
    )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()