# src/hybrid_retriever.py

from typing import List, Dict, Any
from collections import defaultdict

from bm25_retriever import (
    BM25Retriever,
    load_chunks as load_bm25_chunks,
    CHUNKS_FILE,
)

from dense_retriever import DenseRetriever


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DEFAULT_TOP_K = 5

# Number of candidates retrieved from each retriever
CANDIDATE_K = 20

# Standard RRF constant
RRF_K = 60


# ---------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------

class HybridRetriever:

    def __init__(
        self,
        chunks: List[Dict[str, Any]]
    ):
        """
        Initialise BM25 and Dense retrievers.
        """

        self.chunks = chunks

        print("\nInitialising BM25 Retriever...")

        self.bm25 = BM25Retriever(
            chunks
        )

        print("\nInitialising Dense Retriever...")

        self.dense = DenseRetriever(
            chunks
        )

        print(
            "\nHybrid Retriever ready."
        )


    # -----------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and Dense rankings using
        Reciprocal Rank Fusion (RRF).

        Formula:

            score = 1 / (RRF_K + rank)

        Raw BM25 and Dense scores are not directly
        compared because they use different scales.
        """

        fused_scores = defaultdict(float)

        result_lookup = {}

        bm25_rank_lookup = {}

        dense_rank_lookup = {}


        # -------------------------------------------------
        # BM25 rankings
        # -------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1
        ):

            chunk_id = result[
                "chunk_id"
            ]

            fused_scores[
                chunk_id
            ] += (
                1.0 /
                (RRF_K + rank)
            )

            result_lookup[
                chunk_id
            ] = result.copy()

            bm25_rank_lookup[
                chunk_id
            ] = rank


        # -------------------------------------------------
        # Dense rankings
        # -------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1
        ):

            chunk_id = result[
                "chunk_id"
            ]

            fused_scores[
                chunk_id
            ] += (
                1.0 /
                (RRF_K + rank)
            )

            if chunk_id not in result_lookup:

                result_lookup[
                    chunk_id
                ] = result.copy()

            dense_rank_lookup[
                chunk_id
            ] = rank


        # -------------------------------------------------
        # Create final result objects
        # -------------------------------------------------

        combined_results = []

        for chunk_id, rrf_score in (
            fused_scores.items()
        ):

            result = result_lookup[
                chunk_id
            ].copy()

            result[
                "rrf_score"
            ] = float(
                rrf_score
            )

            result[
                "bm25_rank"
            ] = bm25_rank_lookup.get(
                chunk_id
            )

            result[
                "dense_rank"
            ] = dense_rank_lookup.get(
                chunk_id
            )

            combined_results.append(
                result
            )


        # Highest RRF score first
        combined_results.sort(
            key=lambda x: x[
                "rrf_score"
            ],
            reverse=True
        )

        return combined_results


    # -----------------------------------------------------
    # Search one company
    # -----------------------------------------------------

    def _search_single_company(
        self,
        query: str,
        company: str
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search for one company.
        """

        bm25_results = (
            self.bm25.search(
                query=query,
                top_k=CANDIDATE_K,
                company=company
            )
        )

        dense_results = (
            self.dense.search(
                query=query,
                top_k=CANDIDATE_K,
                company=company
            )
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                bm25_results,
                dense_results
            )
        )

        return fused_results


    # -----------------------------------------------------
    # Main Search
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        company: str = "BOTH"
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search.

        For CBA or NAB:
            Run normal hybrid retrieval.

        For BOTH:
            Search each bank separately first so
            one company's chunks do not completely
            dominate the retrieval results.
        """

        query = query.strip()

        if not query:
            return []

        company = company.upper()


        # -------------------------------------------------
        # Single company
        # -------------------------------------------------

        if company in {
            "CBA",
            "NAB"
        }:

            results = (
                self._search_single_company(
                    query,
                    company
                )
            )

            return results[
                :top_k
            ]


        # -------------------------------------------------
        # BOTH companies
        # -------------------------------------------------

        if company == "BOTH":

            cba_results = (
                self._search_single_company(
                    query,
                    "CBA"
                )
            )

            nab_results = (
                self._search_single_company(
                    query,
                    "NAB"
                )
            )


            # Combine results from both banks
            combined = (
                cba_results
                +
                nab_results
            )


            # Sort by RRF score
            combined.sort(
                key=lambda x: x[
                    "rrf_score"
                ],
                reverse=True
            )


            # -------------------------------------------------
            # Ensure both companies are represented
            # -------------------------------------------------

            final_results = []

            seen_chunk_ids = set()


            # First take best CBA result
            best_cba = next(
                (
                    result
                    for result in combined
                    if result[
                        "ticker"
                    ] == "CBA"
                ),
                None
            )

            if best_cba:

                final_results.append(
                    best_cba
                )

                seen_chunk_ids.add(
                    best_cba[
                        "chunk_id"
                    ]
                )


            # Then best NAB result
            best_nab = next(
                (
                    result
                    for result in combined
                    if result[
                        "ticker"
                    ] == "NAB"
                ),
                None
            )

            if best_nab:

                final_results.append(
                    best_nab
                )

                seen_chunk_ids.add(
                    best_nab[
                        "chunk_id"
                    ]
                )


            # Fill remaining slots by score
            for result in combined:

                if (
                    result[
                        "chunk_id"
                    ]
                    in seen_chunk_ids
                ):
                    continue

                final_results.append(
                    result
                )

                seen_chunk_ids.add(
                    result[
                        "chunk_id"
                    ]
                )

                if (
                    len(
                        final_results
                    )
                    >= top_k
                ):
                    break


            return final_results[
                :top_k
            ]


        raise ValueError(
            "Company must be "
            "CBA, NAB, or BOTH."
        )


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

def display_results(
    query: str,
    results: List[Dict[str, Any]]
) -> None:

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
            f"Hybrid RRF Score: "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"BM25 Rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Dense Rank: "
            f"{result.get('dense_rank')}"
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
# Company Selection
# ---------------------------------------------------------

def ask_company() -> str:

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
# Interactive Search
# ---------------------------------------------------------

def interactive_search(
    retriever: HybridRetriever
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "StockLens Hybrid Retriever"
    )

    print(
        "=" * 80
    )

    print(
        "\nBM25 + Dense Search + RRF"
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
        "StockLens Hybrid Retrieval System"
    )

    print(
        "=" * 80
    )


    print(
        f"\nLoading chunks from:\n"
        f"{CHUNKS_FILE}"
    )


    chunks = load_bm25_chunks(
        CHUNKS_FILE
    )


    print(
        f"\nLoaded "
        f"{len(chunks)} usable chunks."
    )


    retriever = HybridRetriever(
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