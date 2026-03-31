"""EPUB text extraction using ebooklib + BeautifulSoup."""
import io
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def parse_epub(file_bytes: bytes) -> str:
    """Extract full text from an EPUB given its raw bytes."""
    book = epub.read_epub(io.BytesIO(file_bytes))
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            chapters.append(text.strip())
    return "\n\n".join(chapters)
