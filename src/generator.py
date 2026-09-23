# src/generator.py

from typing import List, Dict, Any
import os

import ollama

from bm25_retriever import (
    load_chunks,
    CHUNKS_FILE,
)

from hybrid_retriever import HybridRetriever


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DEFAULT_TOP_K = 3

# You can change this later if needed.
# Example:
# qwen2.5:7b
# qwen2.5:3b
# llama3.1:8b
MODEL_NAME = os.getenv(
    "STOCKLENS_MODEL",
    "qwen2.5:7b"
)


# ---------------------------------------------------------
# Build evidence context
# ---------------------------------------------------------

def build_context(
    results: List[Dict[str, Any]]
) -> str:
    """
    Convert retrieved chunks into structured evidence
    for the LLM.

    Each retrieved chunk receives a source label such as:
    [S1], [S2], etc.
    """

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        source_id = f"S{index}"

        company = result.get(
            "company",
            "Unknown"
        )

        ticker = result.get(
            "ticker",
            "Unknown"
        )

        document = result.get(
            "document",
            "Unknown document"
        )

        page = result.get(
            "page",
            "Unknown"
        )

        text = result.get(
            "text",
            ""
        )

        context = f"""
[{source_id}]
Company: {company}
Ticker: {ticker}
Document: {document}
Page: {page}

Evidence:
{text}
"""

        context_parts.append(
            context.strip()
        )

    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------
# System prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are StockLens, an evidence-grounded financial research assistant.

Answer the user's question using ONLY the supplied evidence from official
company reports.

IMPORTANT RULES:

1. If the answer or explanation appears anywhere in the supplied evidence,
   you MUST answer the question.
2. Do NOT say "Insufficient evidence" if a retrieved source directly contains
   the requested fact, number, explanation, risk, or comparison.
3. If the evidence contains multiple versions of a financial metric,
   explain the distinction clearly instead of refusing.
   For example, distinguish:
   - continuing operations
   - total including discontinued operations
   - statutory profit
   - cash profit
4. Use citations such as [S1], [S2], etc.
5. Do not invent information that is not present in the evidence.
6. Keep the answer concise.
7. Preserve financial units and percentages exactly.
8. For comparison questions, discuss each company separately when appropriate.
9. Do not provide buy/sell recommendations or unsupported stock-price predictions.
10. Only use:
   "Insufficient evidence in the available documents."
   when none of the supplied evidence can answer the question.
11. Do not mention irrelevant retrieved sources.
12. Cite only the sources that directly support the answer.
13. Do not explain why unused sources were ignored.

Example:

Question:
What was CBA's statutory net profit in 2025?

Evidence states:
- statutory NPAT from continuing operations was $10,133 million
- total statutory NPAT including discontinued operations was $10,116 million

Correct answer:
CBA reported statutory NPAT of $10,133 million from continuing operations
in FY2025. Including discontinued operations, statutory NPAT was
$10,116 million [S1].

Never refuse when the evidence directly answers the question.
"""

# ---------------------------------------------------------
# Generate answer
# ---------------------------------------------------------

def generate_answer(
    query: str,
    results: List[Dict[str, Any]]
) -> str:
    """
    Generate a grounded answer using Ollama.
    """

    if not results:
        return (
            "Insufficient evidence in the "
            "available documents."
        )

    context = build_context(
        results
    )

    user_prompt = f"""
QUESTION:
{query}

EVIDENCE:
{context}

INSTRUCTIONS:

Answer the question directly from the evidence above.

If a source contains the requested fact or explanation, use it.

If multiple figures appear for the same metric, explain why they differ
instead of refusing.

Cite each factual claim using [S1], [S2], etc.

Only respond with:
"Insufficient evidence in the available documents."
if none of the evidence answers the question.
"""

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": 0.1
            }
        )

        # Compatible with newer Ollama Python clients
        try:
            return response.message.content.strip()

        # Compatibility fallback
        except AttributeError:
            return (
                response["message"]["content"]
                .strip()
            )

    except Exception as exc:

        print(
            "\nERROR: Could not communicate "
            "with Ollama."
        )

        print(
            "\nMake sure:"
        )

        print(
            "1. Ollama is installed."
        )

        print(
            "2. Ollama is running."
        )

        print(
            f"3. Model '{MODEL_NAME}' "
            "has been downloaded."
        )

        print(
            f"\nTechnical error:\n{exc}"
        )

        return (
            "Unable to generate an answer "
            "because the local LLM is unavailable."
        )


# ---------------------------------------------------------
# Display sources
# ---------------------------------------------------------

def display_sources(
    results: List[Dict[str, Any]]
) -> None:
    """
    Display retrieved evidence metadata.

    These citations are generated from retrieval metadata,
    not invented by the LLM.
    """

    print(
        "\n"
        + "-" * 80
    )

    print(
        "SOURCES"
    )

    print(
        "-" * 80
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"[S{index}] "
            f"{result['ticker']} | "
            f"{result['document']} | "
            f"Page {result['page']} | "
            f"Chunk {result['chunk_id']}"
        )


# ---------------------------------------------------------
# Company menu
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
# Interactive StockLens
# ---------------------------------------------------------

def interactive_chat(
    retriever: HybridRetriever
) -> None:

    print(
        "\n"
        + "=" * 80
    )

    print(
        "StockLens RAG Assistant"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLLM: {MODEL_NAME}"
    )

    print(
        "Retrieval: BM25 + Dense + RRF"
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
            "\nAsk a question: "
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


        # ---------------------------------------------
        # Retrieve evidence
        # ---------------------------------------------

        print(
            "\nRetrieving evidence..."
        )

        results = retriever.search(
            query=query,
            top_k=DEFAULT_TOP_K,
            company=company
        )


        # ---------------------------------------------
        # Generate answer
        # ---------------------------------------------

        print(
            "Generating answer..."
        )

        answer = generate_answer(
            query=query,
            results=results
        )


        # ---------------------------------------------
        # Display final result
        # ---------------------------------------------

        print(
            "\n"
            + "=" * 80
        )

        print(
            "STOCKLENS ANSWER"
        )

        print(
            "=" * 80
        )

        print(
            f"\n{answer}"
        )

        display_sources(
            results
        )

        print(
            "=" * 80
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print(
        "=" * 80
    )

    print(
        "StockLens Evidence-Grounded RAG"
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


    # ---------------------------------------------
    # Hybrid Retriever
    # ---------------------------------------------

    retriever = HybridRetriever(
        chunks
    )


    # ---------------------------------------------
    # Start assistant
    # ---------------------------------------------

    interactive_chat(
        retriever
    )


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()