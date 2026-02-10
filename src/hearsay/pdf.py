"""PDF text extraction using PyMuPDF."""

import re
from pathlib import Path

import fitz  # PyMuPDF


# Patterns that indicate a cover/citation page to skip
SKIP_PAGE_PATTERNS = [
    r"You may also like",
    r"To cite this article:",
    r"View the article online for updates",
    r"This content was downloaded from IP address",
]

# Multi-line blocks to remove (applied before line-by-line cleaning)
REMOVE_BLOCKS = [
    # HAL archive header block
    r"HAL Id:.*?(?=From sedimentary|Abstract|Introduction|\n\n[A-Z][a-z])",
    # JSTOR header block
    r"Source:.*?Stable URL:.*?(?=ABSTRACT|Abstract|Introduction)",
    r"JSTOR is a not-for-profit service.*?(?=ABSTRACT|Abstract|[A-Z]{2,})",
    r"Your use of the JSTOR archive.*?(?=\n\n)",
    r"This content downloaded from.*?(?=\n\n|\n[A-Z])",
]

# Patterns to remove from text (line by line)
REMOVE_PATTERNS = [
    r"This content was downloaded from IP address.*?(?=\n|$)",
    r"This content downloaded from.*$",
    r"All use subject to https://about\.jstor\.org/terms\s*$",
    r"^OPEN ACCESS\s*$",
    r"^RECEIVED\s*$",
    r"^REVISED\s*$",
    r"^ACCEPTED FOR PUBLICATION\s*$",
    r"^PUBLISHED\s*$",
    r"^\d{1,2}\s+\w+\s+\d{4}\s*$",  # Dates like "25 September 2022"
    r"^Original content from\s*$",
    r"^this work may be used\s*$",
    r"^under the terms of the\s*$",
    r"^Creative Commons\s*$",
    r"^Attribution \d\.\d licence\.?\s*$",
    r"^Any further distribution\s*$",
    r"^of this work must\s*$",
    r"^maintain attribution to\s*$",
    r"^the author\(s\) and the title\s*$",
    r"^of the work,? journal\s*$",
    r"^citation and DOI\.\s*$",
    r"^https?://doi\.org/.*$",
    r"^https?://\S+\s*$",  # Standalone URLs
    r"^www\.[a-z\-]+\.(org|com|edu)\s*$",  # Website URLs like www.cerf-jcr.org
    r"^Environ\. Res\. Lett\..*$",  # Journal header
    r"©\s*\d{4}\s+The Author\(s\)\..*$",  # Copyright footer
    r"^[A-Z]\s+[A-Z][a-z]+\s+et\s+al\s*$",  # Author page headers like "Y Ma et al"
    r"^IOP Publishing\s*$",
    r"^a\s+r\s+t\s+i\s+c\s+l\s+e\s+i\s+n\s+f\s+o\s*$",  # Spaced "article info"
    r"^s\s+u\s+m\s+m\s+a\s+r\s+y\s*$",  # Spaced "summary"
    r"^Article history:\s*$",
    r"^Received.*revised.*Accepted.*$",
    r"^Available online.*$",
    r"^This manuscript was handled by.*$",
    r"^Keywords:\s*$",
]


def extract_text_raw(pdf_path: Path) -> str:
    """Extract raw text from PDF with minimal cleaning.

    Only fixes ligatures and normalizes whitespace. Use this when
    sending to Claude for AI-based cleaning.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Raw extracted text with minimal cleaning.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    text_parts = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
    finally:
        doc.close()

    text = "\n".join(text_parts)

    # Fix ligatures only
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # Basic whitespace normalization
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_text(pdf_path: Path) -> str:
    """Extract text content from a PDF file with full cleaning.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        RuntimeError: If the PDF cannot be opened or read.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    text_parts = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Skip cover/citation pages
            if _is_skip_page(text):
                continue

            if text.strip():
                text_parts.append(text)
    finally:
        doc.close()

    full_text = "\n".join(text_parts)
    return clean_text(full_text)


def _is_skip_page(text: str) -> bool:
    """Check if a page is a cover/citation page that should be skipped."""
    for pattern in SKIP_PAGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def clean_text(text: str) -> str:
    """Clean extracted PDF text.

    Removes artifacts, rejoins broken lines, and makes text human-readable.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Fix common ligature issues
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)

    # Remove multi-line block patterns first
    for pattern in REMOVE_BLOCKS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    # Remove known line patterns
    for pattern in REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # Remove standalone page numbers
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines BEFORE rejoining (so we have clean single/double breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Rejoin lines broken mid-sentence by PDF column layout
    text = _rejoin_broken_lines(text)

    # Clean up any remaining excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


def _rejoin_broken_lines(text: str) -> str:
    """Rejoin lines that were broken mid-sentence by PDF layout.

    Uses heuristics to determine when lines should be joined vs kept separate.
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look ahead and join continuation lines
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            # Handle empty lines (possible page breaks mid-sentence)
            if not next_line:
                # Look ahead past the empty line
                if i + 2 < len(lines):
                    after_empty = lines[i + 2].strip()
                    # If current line doesn't end sentence and line after empty
                    # starts with lowercase, bridge the gap (page break mid-sentence)
                    if (after_empty and after_empty[0].islower() and
                        not _ends_sentence(line) and not _is_heading(after_empty)):
                        i += 1  # Skip empty line
                        next_line = after_empty
                    else:
                        break
                else:
                    break

            # Don't join if current line is a heading
            if _is_heading(line):
                break

            # Don't join if next line is a heading
            if _is_heading(next_line):
                break

            # Join if current line ends with hyphen (word break)
            if line.endswith("-") and not line.endswith(" -"):
                line = line[:-1] + next_line
                i += 1
                continue

            # Check if this line looks like it ends mid-sentence
            ends_complete = _ends_sentence(line)

            # If line doesn't end a sentence, join with next
            if not ends_complete:
                line = line + " " + next_line
                i += 1
            else:
                break

        result.append(line)
        i += 1

    return "\n".join(result)


def _ends_sentence(line: str) -> bool:
    """Check if a line appears to end a complete sentence or thought."""
    if not line:
        return True

    # Ends with terminal punctuation
    if re.search(r"[.!?]\s*$", line):
        return True

    # Ends with closing parenthesis after citation (common pattern)
    if re.search(r"\d{4}[a-z]?\)\s*\.?\s*$", line):
        return True

    # Ends with colon (often introduces a list or section)
    if line.endswith(":"):
        return True

    # These patterns indicate sentence continues on next line
    continuation_endings = [
        r",$",           # Comma
        r";$",           # Semicolon
        r"\band$",       # "and"
        r"\bor$",        # "or"
        r"\bthe$",       # "the"
        r"\ba$",         # "a"
        r"\ban$",        # "an"
        r"\bof$",        # "of"
        r"\bin$",        # "in"
        r"\bto$",        # "to"
        r"\bfor$",       # "for"
        r"\bwith$",      # "with"
        r"\bby$",        # "by"
        r"\bfrom$",      # "from"
        r"\bthat$",      # "that"
        r"\bwhich$",     # "which"
        r"\bas$",        # "as"
        r"\bat$",        # "at"
        r"\bon$",        # "on"
        r"\bis$",        # "is"
        r"\bare$",       # "are"
        r"\bwas$",       # "was"
        r"\bwere$",      # "were"
        r"\bet$",        # "et" (et al)
        r"\bal$",        # "al" (et al)
        r"\(.*$",        # Open parenthesis not closed
    ]

    for pattern in continuation_endings:
        if re.search(pattern, line, re.IGNORECASE):
            return False

    # If line is short and doesn't end with punctuation, likely continues
    if len(line) < 60 and not re.search(r"[.!?:;,]$", line):
        return False

    # Default: assume it doesn't end (safer for prose)
    return False


def _is_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    if not line:
        return False

    # Numbered sections like "1. Introduction" or "2.1 Methods"
    if re.match(r"^\d+\.?\d*\.?\s+[A-Z]", line):
        return True

    # All caps short lines
    if line.isupper() and len(line) < 60:
        return True

    # Common heading words at start
    heading_words = ["abstract", "introduction", "methods", "results",
                     "discussion", "conclusion", "references", "acknowledgment",
                     "keywords", "summary", "background"]
    first_word = line.split()[0].lower().rstrip(".:") if line.split() else ""
    if first_word in heading_words and len(line) < 40:
        return True

    # Very short lines that start with caps are likely headings
    if len(line) < 30 and line[0].isupper() and not line.endswith(","):
        words = line.split()
        if len(words) <= 4:
            return True

    return False


def extract_figures(pdf_path: Path, output_dir: Path, min_size: int = 10000) -> list[Path]:
    """Extract figures/images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save extracted images.
        min_size: Minimum image size in bytes to include (filters out icons/logos).

    Returns:
        List of paths to extracted image files.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    extracted = []
    figure_num = 1

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Skip small images (likely icons, logos, etc.)
                    if len(image_bytes) < min_size:
                        continue

                    # Determine file extension
                    ext = base_image.get("ext", "png")
                    if ext == "jpeg":
                        ext = "jpg"

                    # Save the image
                    filename = f"figure_{figure_num}.{ext}"
                    output_path = output_dir / filename
                    output_path.write_bytes(image_bytes)

                    extracted.append(output_path)
                    figure_num += 1

                except Exception:
                    # Skip images that can't be extracted
                    continue

    finally:
        doc.close()

    return extracted


def get_text_stats(text: str) -> dict:
    """Get statistics about extracted text.

    Args:
        text: Extracted text.

    Returns:
        Dictionary with word count, character count, etc.
    """
    words = text.split()
    return {
        "characters": len(text),
        "words": len(words),
        "paragraphs": text.count("\n\n") + 1,
    }


def slugify(title: str, max_length: int = 80) -> str:
    """Convert a title to a safe filename slug.

    Args:
        title: The paper title.
        max_length: Maximum length of the slug.

    Returns:
        A filesystem-safe slug.
    """
    # Remove special characters, keep alphanumeric and spaces
    slug = re.sub(r"[^\w\s-]", "", title)
    # Replace spaces with underscores
    slug = re.sub(r"[\s]+", "_", slug)
    # Remove consecutive underscores
    slug = re.sub(r"_+", "_", slug)
    # Truncate and strip
    slug = slug[:max_length].strip("_")
    return slug.lower()


def save_text(text: str, title: str, output_dir: Path) -> Path:
    """Save extracted text to a file in the output directory.

    Args:
        text: The extracted text content.
        title: The paper title (used to generate filename).
        output_dir: Directory to save the file.

    Returns:
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = slugify(title) + ".txt"
    output_path = output_dir / filename

    output_path.write_text(text, encoding="utf-8")
    return output_path


# Quick test when run directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Try to use a paper from Zotero
        from hearsay.zotero import get_papers_in_collection

        papers = get_papers_in_collection("Texas Coast")
        papers_with_pdf = [p for p in papers if p.pdf_path]

        if not papers_with_pdf:
            print("No papers with PDFs found")
            sys.exit(1)

        pdf_path = papers_with_pdf[0].pdf_path
        print(f"Using: {papers_with_pdf[0].title}")
        print(f"PDF: {pdf_path}")
    else:
        pdf_path = Path(sys.argv[1])

    print()
    text = extract_text(pdf_path)
    stats = get_text_stats(text)

    print(f"Stats: {stats['words']} words, {stats['characters']} chars, {stats['paragraphs']} paragraphs")
    print()
    print("First 1000 characters:")
    print("-" * 40)
    print(text[:1000])
