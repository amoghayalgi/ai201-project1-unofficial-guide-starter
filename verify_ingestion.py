"""
Full ingestion verification for Milestone 3.
Loads all Google Review and Reddit files, chunks them, and prints:
  1. Statistics summary
  2. Formatted sample chunks for README.md
  3. Raw sample chunks (no formatting)
"""

from pathlib import Path
from ingest import load_google_reviews, load_reddit_thread, chunk_text

DATA_DIR = Path("data")

# Google Reviews
google_files = [
    "identity_google_reviews.txt",
    "marshall_google_reviews.txt",
    "wahu_google_reviews.txt",
    "venue_google_reviews.txt",
    "grandmarc_google_reviews.txt",
    "ucommons_google_reviews.txt",
    "7west_google_reviews.txt",
]

# Reddit threads
reddit_files = [
    "living_in_dinkytown_reddit.txt",
    "finding_apartment_uofm_reddit.txt",
    "grandmarc_seven_corners_reddit.txt",
    "university_commons_reddit.txt"
]

all_chunks = []

print("=" * 60)
print("GOOGLE REVIEWS")
print("=" * 60)

for fname in google_files:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"[SKIP] {fname} not found")
        continue

    reviews = load_google_reviews(str(fpath))
    chunks = chunk_text(reviews, "google")
    all_chunks.extend(chunks)
    print(f"{fname:40} {len(reviews):3} reviews -> {len(chunks):3} chunks")

google_count = len(all_chunks)

print("\n" + "=" * 60)
print("REDDIT THREADS")
print("=" * 60)

reddit_start = len(all_chunks)

for fname in reddit_files:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"[SKIP] {fname} not found")
        continue

    comments = load_reddit_thread(str(fpath))
    chunks = chunk_text(comments, "reddit")
    all_chunks.extend(chunks)
    print(f"{fname:40} {len(comments):3} blocks -> {len(chunks):3} chunks")

reddit_count = len(all_chunks) - reddit_start

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Google Review chunks:  {google_count:4}")
print(f"Reddit thread chunks:  {reddit_count:4}")
print(f"Total chunks:          {len(all_chunks):4}")
print(f"\nTarget (from planning): ~126 chunks (70 Google + 56 Reddit)")

# Print 5 random chunks for README sample
import random
print("\n" + "=" * 60)
print("5 RANDOM SAMPLE CHUNKS - FORMATTED (for README.md)")
print("=" * 60)

sample = random.sample(all_chunks, min(5, len(all_chunks)))

for i, chunk in enumerate(sample, 1):
    source_name = Path(chunk['source']).name
    print(f'\n### Chunk {i}')
    print(f'**Source:** `{source_name}`')
    if 'building' in chunk:
        print(f'**Building:** {chunk["building"]}')
        print(f'**Rating:** {chunk["rating"]}')
    print(f'**Length:** {len(chunk["text"])} characters')
    print()
    print('```')
    print(chunk['text'])
    print('```')

print("\n" + "=" * 60)
print("5 RANDOM SAMPLE CHUNKS - RAW (no formatting)")
print("=" * 60)

for i, chunk in enumerate(sample, 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print()
