# agent/ingestor.py
import fitz  # PyMuPDF
from typing import List, Dict

def ingest_pdf(file_path: str, paper_id: str) -> List[Dict]:
    """
    Ingests PDF using PyMuPDF, preserving line breaks for the segmenter.
    """
    doc_pages = []
    try:
        doc = fitz.open(file_path)
        for i, page in enumerate(doc):
            # --- PATCH: Get text preserving layout. This keeps newlines. ---
            text = page.get_text("text").strip() 
            doc_pages.append({
                "paper_id": paper_id,
                "page": i + 1,
                "text": text
            })
        print(f"PyMuPDF ingested {len(doc_pages)} pages.")
        return doc_pages
    except Exception as e:
        print(f"Error ingesting PDF with PyMuPDF: {e}")
        return []