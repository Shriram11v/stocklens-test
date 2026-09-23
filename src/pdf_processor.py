# src/pdf_processor.py

from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import json


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "processed_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Company metadata
# ---------------------------------------------------------

COMPANY_MAP = {
    "cba": {
        "company": "Commonwealth Bank of Australia",
        "ticker": "CBA",
    },
    "nab": {
        "company": "National Australia Bank",
        "ticker": "NAB",
    },
}


# ---------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------

def extract_pdf_pages(
    pdf_path: Path,
    company_key: str
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF page-by-page.

    Returns:
        List of dictionaries, where each dictionary contains:
        - company
        - ticker
        - document
        - file_name
        - page
        - text
    """

    if company_key not in COMPANY_MAP:
        raise ValueError(f"Unknown company key: {company_key}")

    company_info = COMPANY_MAP[company_key]

    pages = []

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open PDF: {pdf_path}"
        ) from exc

    for page_index in range(len(document)):
        page = document.load_page(page_index)

        # Extract text
        text = page.get_text("text").strip()

        # Skip blank pages
        if not text:
            continue

        page_record = {
            "company": company_info["company"],
            "ticker": company_info["ticker"],
            "document": pdf_path.stem.replace("_", " ").replace("-", " ").title(),
            "file_name": pdf_path.name,
            "page": page_index + 1,
            "text": text,
        }

        pages.append(page_record)

    document.close()

    return pages


# ---------------------------------------------------------
# Process company folder
# ---------------------------------------------------------

def process_company_folder(company_key: str) -> List[Dict[str, Any]]:
    """
    Process every PDF inside:
        data/<company_key>/
    """

    company_dir = DATA_DIR / company_key

    if not company_dir.exists():
        print(f"[WARNING] Folder not found: {company_dir}")
        return []

    pdf_files = list(company_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {company_dir}")
        return []

    all_pages = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pages = extract_pdf_pages(
            pdf_path=pdf_path,
            company_key=company_key
        )

        print(f"  Extracted {len(pages)} non-empty pages")

        all_pages.extend(pages)

    return all_pages


# ---------------------------------------------------------
# Save processed data
# ---------------------------------------------------------

def save_as_json(
    records: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save extracted records as readable JSON.
    """

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {output_path}")


def save_as_jsonl(
    records: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Save extracted records as JSONL.

    JSONL is useful later for chunking, embeddings,
    retrieval, and evaluation pipelines.
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

    print(f"Saved: {output_path}")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("StockLens PDF Processor")
    print("=" * 60)

    all_records = []

    for company_key in COMPANY_MAP.keys():
        print(f"\nProcessing company: {company_key.upper()}")

        company_records = process_company_folder(company_key)

        all_records.extend(company_records)

        # Save company-specific output
        company_jsonl = (
            OUTPUT_DIR /
            f"{company_key}_pages.jsonl"
        )

        save_as_jsonl(
            company_records,
            company_jsonl
        )

    # Save combined dataset
    combined_json = (
        OUTPUT_DIR /
        "all_pages.json"
    )

    combined_jsonl = (
        OUTPUT_DIR /
        "all_pages.jsonl"
    )

    save_as_json(
        all_records,
        combined_json
    )

    save_as_jsonl(
        all_records,
        combined_jsonl
    )

    print("\n" + "=" * 60)
    print("Processing complete")
    print(f"Total extracted pages: {len(all_records)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()