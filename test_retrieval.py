"""
Milestone 4 — Retrieval Testing
Tests the retrieval pipeline with 3 evaluation queries from planning.md

Run: python test_retrieval.py

Ensure you've run `python embed.py` first to populate the ChromaDB collection.
"""

from embed import retrieve


# Evaluation queries from planning.md
EVALUATION_QUERIES = [
    {
        "id": 1,
        "query": "What do students say about maintenance responsiveness at The Venue Dinkytown?",
        "expected_answer": "Feedback is mixed, but it leans toward generally fast and responsive maintenance"
    },
    {
        "id": 2,
        "query": "Is Wahu considered worth the price?",
        "expected_answer": "Most residents do not feel WaHu is worth the price. Amenities valued by some, but price-to-quality ratio criticized"
    },
    {
        "id": 3,
        "query": "How do students describe safety in Dinkytown?",
        "expected_answer": "Students generally feel Dinkytown is safe enough to live in, as long as you use basic city awareness, especially at night"
    },
]


def test_query(query_dict: dict) -> None:
    """
    Runs a single evaluation query and prints the top-5 results.

    Args:
        query_dict: Dict with keys 'id', 'query', 'expected_answer'
    """
    print("\n" + "=" * 80)
    print(f"QUERY {query_dict['id']}")
    print("=" * 80)
    print(f"Question: {query_dict['query']}")
    print(f"Expected: {query_dict['expected_answer']}")
    print("\n" + "-" * 80)
    print("TOP 5 RETRIEVED CHUNKS:")
    print("-" * 80)

    results = retrieve(query_dict['query'], k=5)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Context: {result['context']}")
        print(f"    Source: {result['source']}")
        print(f"    Chunk Index: {result['chunk_index']}")
        print(f"    Distance: {result['distance']:.4f}")
        print(f"    Text:")
        # Print full text for manual inspection (wrapped at 80 chars for readability)
        text_lines = result['text'].split('\n')
        for line in text_lines:
            if len(line) <= 76:
                print(f"      {line}")
            else:
                # Wrap long lines
                words = line.split()
                current_line = "      "
                for word in words:
                    if len(current_line) + len(word) + 1 <= 80:
                        current_line += word + " "
                    else:
                        print(current_line.rstrip())
                        current_line = "        " + word + " "
                if current_line.strip():
                    print(current_line.rstrip())

    print("\n" + "=" * 80)


def main():
    """Runs all evaluation queries and prints results."""
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION TEST")
    print("=" * 80)
    print(f"Testing {len(EVALUATION_QUERIES)} queries with top-k=5")
    print("Distance metric: L2 (Euclidean) — lower = more similar")
    print("=" * 80)

    for query_dict in EVALUATION_QUERIES:
        test_query(query_dict)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the retrieved chunks for each query")
    print("2. Judge accuracy: accurate / partially accurate / inaccurate")
    print("3. Note distance scores (typically < 0.5 = strong match, > 1.0 = weak)")
    print("4. Document findings in your Milestone 4 README section")
    print("=" * 80)


if __name__ == "__main__":
    main()
