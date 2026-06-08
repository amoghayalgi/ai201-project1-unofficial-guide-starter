"""
Milestone 3 — Document Ingestion
Loads Google Review .txt files and Reddit thread sources (URL or .txt),
then chunks them according to source type.

Library choices:
  - praw: Reddit API wrapper (used only when source is a URL, not a local file)
  - langchain_text_splitters: RecursiveCharacterTextSplitter for Reddit overlap chunking
  - re / pathlib: stdlib only for Google Reviews (no extra deps needed)

Install: pip install praw langchain-text-splitters
"""

import re
import os
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Google Reviews
# ---------------------------------------------------------------------------

def load_google_reviews(filepath: str) -> list[dict]:
    """
    Reads a Google Review .txt file and returns one dict per review.

    Expected file format (from the data/ files):
        Apartment: <Building Name>
        Source: <Source Name or Link>
        ---

        Review 1
        Rating: X/5
        <review body>

        ---

        Review 2
        ...

    Returns a list of dicts:
        {
            "text":    "<Source + Building + Rating + body text>",
            "source":  "<filepath>",
            "building": "<building name parsed from file header>",
            "rating":  "<X/5 string>",
        }

    Flag: If your files use a different separator than "---" on its own line,
    update REVIEW_SEPARATOR below.
    """
    REVIEW_SEPARATOR = re.compile(r"^\s*---\s*$", re.MULTILINE)

    text = Path(filepath).read_text(encoding="utf-8")

    # Pull building name and source link from header
    building = ""
    source_link = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Apartment:"):
            building = line.removeprefix("Apartment:").strip()
        elif line.startswith("Source:"):
            source_link = line.removeprefix("Source:").strip()
        if building and source_link:
            break

    raw_blocks = REVIEW_SEPARATOR.split(text)

    reviews = []
    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        # Skip the header block (contains Apartment: and Source: but no Review N)
        if not re.search(r"^Review\s+\d+", block, re.MULTILINE):
            continue

        # Drop the "Review N" header line — it adds no semantic content
        lines = block.splitlines()
        body_lines = []
        rating = ""
        skip_header = True

        for line in lines:
            stripped = line.strip()
            # Skip the "Apartment: X" and "Source: X" lines if they appear inside a block
            if stripped.startswith("Apartment:") or stripped.startswith("Source:"):
                continue
            if skip_header and re.match(r"^Review\s+\d+\s*$", stripped):
                skip_header = False
                continue
            if stripped.startswith("Rating:"):
                rating = stripped.removeprefix("Rating:").strip()
                continue
            body_lines.append(line)

        body = "\n".join(body_lines).strip()
        if not body:
            continue

        # Prepend source, building, and rating so the embedding captures them
        # Source line is added first for easy extraction in generate_answer()
        full_text = f"Source: {source_link}\nBuilding: {building}\nRating: {rating}\n{body}"

        reviews.append({
            "text":     full_text,
            "source":   str(filepath),
            "building": building,
            "rating":   rating,
        })

    return reviews


# ---------------------------------------------------------------------------
# Reddit Threads
# ---------------------------------------------------------------------------

# Thread context mapping: filename → contextual label prepended to chunks
# This helps embeddings understand which building/topic each comment discusses
THREAD_CONTEXT = {
    "grandmarc_seven_corners_reddit.txt": "Grandmarc Seven Corners",
    "university_commons_reddit.txt": "University Commons",
    "living_in_dinkytown_reddit.txt": "Dinkytown Safety",
    "finding_apartment_uofm_reddit.txt": "Apartment Hunting",
}

def load_reddit_thread(filepath: str) -> list[dict]:
    """
    Loads a Reddit thread from a local .txt file.

    Expected format:
        Source: <URL>
        <Thread title>
        <Thread body>

        ---

        Comment 1
        <comment body>

        - Reply 1
        <reply body>

        ---

        Comment 2
        ...

    Comment blocks are separated by "---" on its own line.
    Each block (thread post or comment subtree) becomes one document.
    "Comment N" headers are automatically stripped (no semantic value).

    Returns a list of dicts:
        {
            "text":   "<comment/subtree text with Source line>",
            "source": "<filepath>",
        }
    """
    BLOCK_SEPARATOR = re.compile(r"^\s*---\s*$", re.MULTILINE)
    COMMENT_HEADER = re.compile(r"^Comment\s+\d+\s*$", re.MULTILINE)

    text = Path(filepath).read_text(encoding="utf-8")

    # Extract source link from first line
    source_link = ""
    lines = text.split('\n')
    if lines and lines[0].strip().startswith("Source:"):
        source_link = lines[0].strip().replace("Source:", "").strip()
        # Remove source line from text before splitting into blocks
        text = '\n'.join(lines[1:])

    blocks = BLOCK_SEPARATOR.split(text)

    # Get context label for this thread (if mapped)
    filename = Path(filepath).name
    context = THREAD_CONTEXT.get(filename, "")

    chunks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Strip "Comment N" header from the beginning of the block
        block = COMMENT_HEADER.sub("", block, count=1).strip()

        if not block:
            continue

        # Prepend source link and thread context so embeddings capture them
        # Source line is added first for easy extraction in generate_answer()
        if source_link and context:
            full_text = f"Source: {source_link}\nDiscussion: {context}\n{block}"
        elif context:
            full_text = f"Discussion: {context}\n{block}"
        elif source_link:
            full_text = f"Source: {source_link}\n{block}"
        else:
            full_text = block

        chunks.append({
            "text":   full_text,
            "source": str(filepath),
            "context": context,  # Store as metadata for attribution
        })

    return chunks


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(docs: list[dict], source_type: str) -> list[dict]:
    """
    Applies the correct chunking strategy by source_type.

    "google" — 1 review per chunk, no overlap.
               docs are already pre-split by load_google_reviews(), so this
               is a pass-through that enforces a max character length as a
               safety net (reviews longer than ~2 000 chars get split with
               a 0-token overlap so no content is lost).

    "reddit" — 1 comment subtree per chunk with 10-20% overlap on long
               threads.  Uses LangChain's RecursiveCharacterTextSplitter
               so that split points fall at paragraph/sentence boundaries
               rather than mid-word.

    Returns a list of chunk dicts:
        {
            "text":    "<chunk text>",
            "source":  "<source filename or URL>",
            ... (any extra keys from the input dict are preserved)
        }
    """
    if source_type == "google":
        return _chunk_google(docs)
    elif source_type == "reddit":
        return _chunk_reddit(docs)
    else:
        raise ValueError(f"Unknown source_type '{source_type}'. Use 'google' or 'reddit'.")


def _chunk_google(docs: list[dict]) -> list[dict]:
    """
    Pass-through for Google reviews (1 review = 1 chunk).
    Falls back to LangChain splitting only if a review exceeds 2 000 chars.
    """
    HARD_LIMIT = 2000  # characters; all-MiniLM-L6-v2 has a 256-token window
    OVERLAP = 200      # ~10% overlap to preserve context at chunk boundaries

    chunks = []
    for doc in docs:
        if len(doc["text"]) <= HARD_LIMIT:
            chunks.append(doc)
        else:
            # Rare: a very long review — split with 10% overlap to preserve context
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError as e:
                raise ImportError(
                    "langchain-text-splitters is required. "
                    "Run: pip install langchain-text-splitters"
                ) from e

            # Extract just the review body (strip the prepended metadata)
            # Format: "Source: X\nBuilding: Y\nRating: Z\n<body>"
            lines = doc["text"].split("\n")
            body_start = 0
            source_line = ""
            for i, line in enumerate(lines):
                if line.startswith("Source:"):
                    source_line = line
                    body_start = i + 1
                elif line.startswith("Building:") or line.startswith("Rating:"):
                    body_start = i + 1
                else:
                    break

            # Join everything after the metadata lines
            body = "\n".join(lines[body_start:])

            # Split the body text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=HARD_LIMIT - 150,  # leave room for metadata (including source)
                chunk_overlap=OVERLAP,
                separators=["\n\n", "\n", ". ", " "],
            )

            # Prepend metadata to EACH split chunk (including source)
            building = doc.get("building", "")
            rating = doc.get("rating", "")
            metadata_prefix = f"{source_line}\nBuilding: {building}\nRating: {rating}\n" if source_line else f"Building: {building}\nRating: {rating}\n"

            for part in splitter.split_text(body):
                chunks.append({
                    **doc,
                    "text": metadata_prefix + part
                })

    return chunks


def _chunk_reddit(docs: list[dict]) -> list[dict]:
    """
    Chunks Reddit comment subtrees with 10-20% overlap (15% target).
    Subtrees that are already short enough are kept as-is.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise ImportError(
            "langchain-text-splitters is required. "
            "Run: pip install langchain-text-splitters"
        ) from e

    CHUNK_SIZE = 800       # characters — fits comfortably in 256-token window
    OVERLAP = 120          # ~15% of 800

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - 50,  # leave room for "Discussion: X\n" prefix
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in docs:
        if len(doc["text"]) <= CHUNK_SIZE:
            chunks.append(doc)
        else:
            # Extract "Source: X\n" and "Discussion: Y\n" prefixes if present
            text = doc["text"]
            prefix = ""
            body = text

            # Check for Source line and Discussion line
            remaining_text = text
            if text.startswith("Source: "):
                lines = text.split("\n", 1)
                if len(lines) == 2:
                    prefix = lines[0] + "\n"  # "Source: X\n"
                    remaining_text = lines[1]

            if remaining_text.startswith("Discussion: "):
                lines = remaining_text.split("\n", 1)
                if len(lines) == 2:
                    prefix += lines[0] + "\n"  # "Source: X\nDiscussion: Y\n"
                    body = lines[1]             # rest of the text
                else:
                    body = remaining_text
            else:
                body = remaining_text

            # Split the body and prepend metadata to each chunk
            for part in splitter.split_text(body):
                chunks.append({
                    **doc,
                    "text": prefix + part
                })

    return chunks
