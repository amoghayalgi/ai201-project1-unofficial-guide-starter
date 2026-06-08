"""
Milestone 6 — Final Evaluation
Runs all 5 evaluation queries and documents results for README.

Run: python run_evaluation.py
"""

from app import answer_question
from embed import retrieve


# 5 Evaluation Queries
EVALUATION_QUERIES = [
    {
        "id": 1,
        "query": "What do students say about maintenance responsiveness at The Venue Dinkytown?",
        "expected": "Feedback is mixed, but it leans toward generally fast and responsive maintenance",
        "should_work": True
    },
    {
        "id": 2,
        "query": "Is Wahu considered worth the price?",
        "expected": "Most residents do not feel WaHu is worth the price. Amenities valued by some, but price-to-quality ratio criticized",
        "should_work": True
    },
    {
        "id": 3,
        "query": "How do students describe safety in Dinkytown?",
        "expected": "Students generally feel Dinkytown is safe enough to live in, as long as you use basic city awareness, especially at night",
        "should_work": True
    },
    {
        "id": 4,
        "query": "What are the main complaints about Grandmarc Seven Corners?",
        "expected": "Management unresponsiveness, package security, lease experience issues, WiFi problems",
        "should_work": True
    },
    {
        "id": 5,
        "query": "What is the monthly rent for a 2-bedroom apartment at Venue Dinkytown?",
        "expected": "Should refuse or give inaccurate answer - rent prices not in review corpus",
        "should_work": False  # FAILURE CASE
    },
]


def evaluate_query(query_dict: dict) -> None:
    """Runs one evaluation query and prints formatted results."""
    print("\n" + "=" * 80)
    print(f"QUERY {query_dict['id']}: {'EXPECTED TO WORK' if query_dict['should_work'] else 'FAILURE CASE'}")
    print("=" * 80)
    print(f"Question: {query_dict['query']}")
    print(f"Expected: {query_dict['expected']}")

    print("\n" + "-" * 80)
    print("RETRIEVAL")
    print("-" * 80)

    # Retrieve chunks
    chunks = retrieve(query_dict['query'], k=5)

    print(f"\nTop 5 retrieved chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] Context: {chunk['context']}")
        print(f"      Distance: {chunk['distance']:.4f}")
        # Extract source line from chunk text
        source_line = ""
        for line in chunk['text'].split('\n'):
            if line.startswith('Source:'):
                source_line = line.replace('Source:', '').strip()
                break
        if source_line:
            print(f"      Source: {source_line}")

    print("\n" + "-" * 80)
    print("GENERATION")
    print("-" * 80)

    # Generate answer
    answer, sources = answer_question(query_dict['query'])

    print(f"\nGenerated Answer:")
    print(answer)

    print(f"\nSources:")
    for source in sources.split('\n'):
        print(source)

    print("\n" + "-" * 80)
    print("ACCURACY JUDGMENT")
    print("-" * 80)

    if not query_dict['should_work']:
        print("\nThis is the FAILURE CASE.")
        print("Analysis: Check if the system:")
        print("  - Retrieved irrelevant chunks (high distance scores)")
        print("  - Generated an answer despite lack of relevant information")
        print("  - Or correctly refused to answer")
    else:
        print("\nJudgment: [ ] Accurate  [ ] Partially Accurate  [ ] Inaccurate")
        print("(Fill in manually after reviewing the answer)")

    print("=" * 80)


def main():
    """Runs all 5 evaluation queries."""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION - MILESTONE 6")
    print("=" * 80)
    print("Running 5 evaluation queries...")
    print("Results will be documented in README.md")
    print("=" * 80)

    for query_dict in EVALUATION_QUERIES:
        try:
            evaluate_query(query_dict)
        except Exception as e:
            print(f"\nERROR evaluating query {query_dict['id']}: {e}")
            continue

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Fill in accuracy judgments (Accurate / Partially Accurate / Inaccurate)")
    print("3. Analyze the failure case (Query 5)")
    print("4. Copy results to README.md Evaluation section")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
