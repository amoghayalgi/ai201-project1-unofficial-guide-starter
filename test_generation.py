"""
Milestone 5 — Generation Testing
End-to-end tests for the RAG pipeline (retrieve → generate).

Tests 3 queries:
  1. In-scope: Venue maintenance (expected: mixed feedback)
  2. In-scope: Wahu price value (expected: mostly negative)
  3. Out-of-scope: Restaurant recommendation (expected: refusal)

Run: python test_generation.py
"""

from embed import retrieve
from generate import generate_answer


# Test queries
TEST_QUERIES = [
    {
        "id": 1,
        "query": "What do students say about maintenance responsiveness at The Venue Dinkytown?",
        "expected": "Mixed feedback — some report fast service, others cite multi-week delays",
        "scope": "in-scope"
    },
    {
        "id": 2,
        "query": "Is Wahu considered worth the price?",
        "expected": "Mostly negative on price-to-quality ratio, management issues cited frequently",
        "scope": "in-scope"
    },
    {
        "id": 3,
        "query": "What is the best restaurant near campus?",
        "expected": "Should refuse — no restaurant information in housing review corpus",
        "scope": "out-of-scope"
    },
]


def test_query(query_dict: dict) -> None:
    """
    Runs one end-to-end RAG test and prints results.

    Args:
        query_dict: Dict with keys 'id', 'query', 'expected', 'scope'
    """
    print("\n" + "=" * 80)
    print(f"TEST {query_dict['id']}: {query_dict['scope'].upper()}")
    print("=" * 80)
    print(f"Query: {query_dict['query']}")
    print(f"Expected: {query_dict['expected']}")
    print("\n" + "-" * 80)
    print("RETRIEVAL")
    print("-" * 80)

    # Step 1: Retrieve
    chunks = retrieve(query_dict['query'], k=5)

    print(f"\nTop 5 chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] Context: {chunk['context']}")
        print(f"      Distance: {chunk['distance']:.4f}")
        print(f"      Preview: {chunk['text'][:100]}...")

    print("\n" + "-" * 80)
    print("GENERATION")
    print("-" * 80)

    # Step 2: Generate
    result = generate_answer(query_dict['query'], chunks)

    print(f"\nAnswer:")
    print(result['answer'])

    print(f"\nSources ({len(result['sources'])} unique):")
    for source in result['sources']:
        print(f"  • {source}")

    # Check for refusal on out-of-scope query
    if query_dict['scope'] == "out-of-scope":
        refusal_phrase = "I don't have enough information"
        if refusal_phrase.lower() in result['answer'].lower():
            print(f"\n✓ PASS: Correctly refused out-of-scope query")
        else:
            print(f"\n✗ FAIL: Did not refuse out-of-scope query")
            print(f"   Expected refusal phrase: '{refusal_phrase}'")

    print("=" * 80)


def main():
    """Runs all end-to-end generation tests."""
    print("\n" + "=" * 80)
    print("END-TO-END GENERATION TEST SUITE")
    print("=" * 80)
    print(f"Testing {len(TEST_QUERIES)} queries (2 in-scope, 1 out-of-scope)")
    print("Model: llama-3.3-70b-versatile via Groq API")
    print("Grounding: Strict (allow synthesis, require citations, refuse off-topic)")
    print("=" * 80)

    for query_dict in TEST_QUERIES:
        test_query(query_dict)

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the generated answers for accuracy and grounding")
    print("2. Verify in-scope queries (1-2) provide helpful, cited answers")
    print("3. Verify out-of-scope query (3) correctly refuses")
    print("4. Document findings in your Milestone 5 README section")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
