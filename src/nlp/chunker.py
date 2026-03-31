"""Scene/paragraph-aware text chunker with overlap for NLP context preservation."""
from typing import List, Tuple


CHUNK_TOKEN_LIMIT = 6000  # ~28 chunks for a full novel instead of 83
OVERLAP_TOKENS = 400       # trailing tokens to prepend to next chunk


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def chunk_text(text: str) -> List[Tuple[str, str]]:
    """
    Split cleaned text into chunks suitable for LLM prompting.

    Returns a list of (chunk_text, overlap_prefix) tuples where
    overlap_prefix is the last ~200 tokens of the previous chunk,
    helping the LLM maintain speaker context across boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[Tuple[str, str]] = []
    current_paragraphs: List[str] = []
    current_tokens = 0
    previous_overlap = ""

    for para in paragraphs:
        para_tokens = _approx_tokens(para)

        if current_tokens + para_tokens > CHUNK_TOKEN_LIMIT and current_paragraphs:
            chunk_text_str = "\n\n".join(current_paragraphs)
            chunks.append((chunk_text_str, previous_overlap))

            # Build overlap from the tail of this chunk
            tail = chunk_text_str[-(OVERLAP_TOKENS * 4):]
            previous_overlap = tail

            current_paragraphs = [para]
            current_tokens = para_tokens
        else:
            current_paragraphs.append(para)
            current_tokens += para_tokens

    # Last chunk
    if current_paragraphs:
        chunks.append(("\n\n".join(current_paragraphs), previous_overlap))

    return chunks
