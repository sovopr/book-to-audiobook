"""PDF text extraction using PyMuPDF (fitz)."""
import fitz  # PyMuPDF


def parse_pdf(file_bytes: bytes) -> str:
    """Extract full text from a PDF given its raw bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)
