# src/dense_retriever.py

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"

CHUNKS_FILE = (
    PROCESSED_DATA_DIR
    / "all_chunks.jsonl"
)

FAISS_INDEX_FILE = (
    PROCESSED_DATA_DIR
    / "dense_faiss.index"
)

INDEX_META_FILE = (
    PROCESSED_DATA_DIR
    / "dense_index_meta.json"
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 5

MIN_WORD_COUNT = 20

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
    Load chunks created by chunking.py.

    Chunks smaller than MIN_WORD_COUNT are ignored
    so that Dense and BM25 retrieval use approximately
    the same usable document collection.
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
                    f"Invalid JSON on line {line_number}"
                ) from exc

            if record.get(
                "word_count",
                0
            ) < MIN_WORD_COUNT:
                continue

            chunks.append(record)

    return chunks


# ---------------------------------------------------------
# Index metadata
# ---------------------------------------------------------

def get_current_index_metadata(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Metadata used to determine whether the saved FAISS
    index still matches the current chunks.
    """

    return {
        "model_name": MODEL_NAME,
        "chunk_count": len(chunks),
        "min_word_count": MIN_WORD_COUNT,
        "chunks_file_mtime_ns": (
            CHUNKS_FILE.stat().st_mtime_ns
        ),
    }


def index_is_current(
    chunks: List[Dict[str, Any]]
) -> bool:
    """
    Check whether an existing FAISS index can safely
    be reused.
    """

    if not FAISS_INDEX_FILE.exists():
        return False

    if not INDEX_META_FILE.exists():
        return False

    try:

        with open(
            INDEX_META_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            saved_metadata = json.load(file)

    except Exception:
        return False

    current_metadata = get_current_index_metadata(
        chunks
    )

    return (
        saved_metadata.get("model_name")
        == current_metadata["model_name"]

        and saved_metadata.get("chunk_count")
        == current_metadata["chunk_count"]

        and saved_metadata.get("min_word_count")
        == current_metadata["min_word_count"]

        and saved_metadata.get(
            "chunks_file_mtime_ns"
        )
        == current_metadata[
            "chunks_file_mtime_ns"
        ]
    )


def save_index_metadata(
    chunks: List[Dict[str, Any]],
    embedding_dimension: int
) -> None:
    """
    Save information about the generated index.
    """

    metadata = get_current_index_metadata(
        chunks
    )

    metadata[
        "embedding_dimension"
    ] = embedding_dimension

    with open(
        INDEX_META_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


# ---------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------

class DenseRetriever:

    def __init__(
        self,
        chunks: List[Dict[str, Any]]
    ):
        """
        Initialise embedding model and FAISS index.
        """

        self.chunks = chunks

        print(
            f"\nLoading embedding model:\n"
            f"{MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        print(
            "Embedding model loaded."
        )

        self.index = self._load_or_build_index()


    # -----------------------------------------------------
    # Build FAISS index
    # -----------------------------------------------------

    def _build_index(self):
        """
        Convert every chunk into an embedding and
        build a FAISS index.
        """

        print(
            "\nCreating embeddings for "
            f"{len(self.chunks)} chunks..."
        )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        embedding_dimension = (
            embeddings.shape[1]
        )

        print(
            "\nEmbedding dimension:",
            embedding_dimension
        )

        # Because embeddings are normalized,
        # inner product behaves like cosine similarity.
        index = faiss.IndexFlatIP(
            embedding_dimension
        )

        index.add(
            embeddings
        )

        print(
            f"FAISS index contains "
            f"{index.ntotal} vectors."
        )

        # Save index so embeddings do not need to be
        # generated every time the program starts.
        faiss.write_index(
            index,
            str(FAISS_INDEX_FILE)
        )

        save_index_metadata(
            self.chunks,
            embedding_dimension
        )

        print(
            f"\nFAISS index saved to:\n"
            f"{FAISS_INDEX_FILE}"
        )

        return index


    # -----------------------------------------------------
    # Load or build index
    # -----------------------------------------------------

    def _load_or_build_index(self):
        """
        Load saved FAISS index if it matches the current
        chunk dataset. Otherwise rebuild it.
        """

        if index_is_current(
            self.chunks
        ):

            print(
                "\nExisting FAISS index found."
            )

            print(
                "Loading saved index..."
            )

            index = faiss.read_index(
                str(FAISS_INDEX_FILE)
            )

            if (
                index.ntotal
                != len(self.chunks)
            ):

                print(
                    "Index size does not match "
                    "current chunks."
                )

                print(
                    "Rebuilding index..."
                )

                return self._build_index()

            print(
                f"Loaded FAISS index with "
                f"{index.ntotal} vectors."
            )

            return index

        print(
            "\nNo current FAISS index found."
        )

        print(
            "Building a new dense index..."
        )

        return self._build_index()


    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS vector index.

        Args:
            query:
                User's financial question.

            top_k:
                Number of results to return.

            company:
                CBA
                NAB
                BOTH
                None

        Returns:
            Ranked dense retrieval results.
        """

        query = query.strip()

        if not query:
            return []

        # Convert question into vector
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        # Search all chunks.
        #
        # Dataset is very small (~1,000 chunks), so
        # searching all vectors makes company filtering
        # completely reliable.
        search_k = len(
            self.chunks
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                search_k
            )
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0]
        ):

            # FAISS sometimes uses -1 when no
            # result exists.
            if index_position == -1:
                continue

            chunk = self.chunks[
                int(index_position)
            ]

            ticker = (
                chunk
                .get(
                    "ticker",
                    ""
                )
                .upper()
            )

            # Company filter
            if (
                company
                and company.upper()
                != "BOTH"
                and ticker
                != company.upper()
            ):
                continue

            result = {
                "score": float(score),
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

            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------
# Display search results
# ---------------------------------------------------------

def display_results(
    query: str,
    results: List[Dict[str, Any]]
) -> None:
    """
    Display dense retrieval results.
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
            f"Similarity Score: "
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
# Company menu
# ---------------------------------------------------------

def ask_company() -> str:
    """
    Ask which company should be searched.
    """

    while True:

        print(
            "\nChoose company:"
        )

        print(
            "1. CBA"
        )

        print(
            "2. NAB"
        )

        print(
            "3. BOTH"
        )

        print(
            "4. EXIT"
        )

        choice = input(
            "\nEnter 1, 2, 3, or 4: "
        ).strip().lower()

        if choice == "1":
            return "CBA"

        if choice == "2":
            return "NAB"

        if choice == "3":
            return "BOTH"

        if choice in {
            "4",
            "exit",
            "quit",
            "q"
        }:
            return "EXIT"

        print(
            "Invalid choice. "
            "Please try again."
        )


# ---------------------------------------------------------
# Interactive search
# ---------------------------------------------------------

def interactive_search(
    retriever: DenseRetriever
) -> None:
    """
    Interactive command-line Dense RAG search.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "StockLens Dense Retriever"
    )

    print(
        "=" * 80
    )

    print(
        "\nUsing model:"
    )

    print(
        MODEL_NAME
    )

    while True:

        company = ask_company()

        if company == "EXIT":

            print(
                "\nExiting StockLens."
            )

            break

        print(
            f"\nSelected: {company}"
        )

        query = input(
            "\nAsk a financial question: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nExiting StockLens."
            )

            break

        if not query:

            print(
                "Please enter a question."
            )

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
        "StockLens Dense Retrieval System"
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

    retriever = DenseRetriever(
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