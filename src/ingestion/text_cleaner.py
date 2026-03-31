"""Clean extracted raw text: remove page numbers, headers, junk characters."""
import re
from collections import Counter


def _remove_repeated_short_lines(text: str, threshold: int = 3) -> str:
    """Remove lines that appear many times and are short (running headers/footers)."""
    lines = text.split("\n")
    line_counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) < 60)
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    cleaned = [l for l in lines if l.strip() not in repeated]
    return "\n".join(cleaned)


def clean_text(raw_text: str) -> str:
    """Full cleaning pipeline for extracted book text."""
    text = raw_text

    # Remove standalone page numbers (line is just digits, possibly with spaces)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove repeated short header/footer lines
    text = _remove_repeated_short_lines(text)

    # Strip non-printable characters (except newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)

    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()
