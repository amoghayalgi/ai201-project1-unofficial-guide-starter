"""
Milestone 5 — Grounded Generation
Generates answers using Groq LLM with strict grounding to retrieved context.

Dependencies:
  - groq: Groq API client
  - python-dotenv: Load GROQ_API_KEY from .env file
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Groq client
_CLIENT = None

def _get_client():
    """Lazy-load Groq client with API key from .env"""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. "
                "Add it to your .env file:\n"
                "GROQ_API_KEY=your_api_key_here"
            )
        _CLIENT = Groq(api_key=api_key)
    return _CLIENT


def build_context(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a numbered context block for the LLM prompt.

    Args:
        chunks: List of retrieved chunk dicts from retrieve()
                Expected keys: 'text', 'source', 'context', 'chunk_index', 'distance'

    Returns:
        Formatted string with numbered chunks, each showing source and context label.

    Example output:
        [1] Source: data\venue_google_reviews.txt | Context: Venue at Dinkytown
        Building: Venue at Dinkytown
        Rating: 1/5
        Severe maintenance and management issues...

        [2] Source: data\wahu_google_reviews.txt | Context: WaHu Student Apartments
        ...
    """
    if not chunks:
        return "[No relevant context retrieved]"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # Format: [N] Source: ... | Context: ...
        header = f"[{i}] Source: {chunk['source']} | Context: {chunk['context']}"
        # Add the chunk text (already formatted with Building/Rating or Discussion prefix)
        text = chunk['text']
        # Combine with blank line separator
        context_parts.append(f"{header}\n{text}")

    return "\n\n".join(context_parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generates a grounded answer using Groq's llama-3.3-70b-versatile.

    Args:
        query: User's question
        chunks: Retrieved chunks from retrieve()

    Returns:
        Dict with keys:
            - 'answer': The generated answer string
            - 'sources': List of unique source file paths cited

    Grounding rules (enforced via system prompt):
        1. Answer only using provided context
        2. Allow synthesis across chunks (don't require word-for-word matches)
        3. Cite sources for every factual claim
        4. Refuse only if chunks are genuinely off-topic
        5. Do NOT refuse just because answer requires light inference

    API notes:
        - Model: llama-3.3-70b-versatile (Groq's fastest 70B model)
        - Temperature: 0.0 (deterministic, reduces hallucination)
        - Max tokens: 512 (enough for detailed answers)
    """
    print(f"[generate_answer] Query: '{query}'")
    print(f"[generate_answer] Retrieved {len(chunks)} chunks")

    # Build the context block
    context = build_context(chunks)

    # System prompt with grounding rules
    system_prompt = """You are a helpful assistant that answers questions about off-campus housing near the University of Minnesota based on student reviews and discussions.

GROUNDING RULES:
1. Answer ONLY using the provided context chunks below. Do not use any external knowledge.

2. Synthesize information across multiple chunks to provide comprehensive answers. This includes:
   - Aggregating feedback about a single building
   - Comparing different buildings when asked
   - Identifying patterns across multiple reviews

3. Do NOT cite sources in your answer text. This includes:
   - No phrases like "According to...", "Students mention...", "Reviews indicate..."
   - No reference numbers like [1], [2], [3] or (Source 1), etc.
   - Just provide the information directly
   The sources will be shown separately to the user.

4. For comparison questions (e.g., "compare X and Y"):
   - If you have chunks from both buildings, provide a structured comparison
   - If you only have chunks from one building, acknowledge this and provide information about the building you have data for, noting that you don't have information about the other
   - Only refuse if you have no relevant chunks for either building

5. If the question is completely unrelated to the retrieved context (e.g., asking about restaurants when all chunks discuss housing), respond with EXACTLY: "I don't have enough information in my sources to answer that question."

6. Do NOT refuse just because:
   - The answer requires synthesis across chunks
   - You need to compare buildings that aren't mentioned together in a single chunk
   - The evidence is mixed or contradictory

7. Be direct and helpful. Acknowledge when evidence is mixed (e.g., "Feedback is mixed — some residents report fast maintenance while others experienced delays").

Provide a clear, well-organized answer WITHOUT source citations."""

    # User prompt with context and query
    user_prompt = f"""CONTEXT:
{context}

QUESTION: {query}

Provide a direct, helpful answer based on the context above. Do NOT include source citations in your answer."""

    # Call Groq API
    client = _get_client()

    print(f"[generate_answer] Calling Groq API (llama-3.3-70b-versatile)...")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,  # Deterministic output, reduces hallucination
        max_tokens=512,   # Enough for detailed answers
    )

    # Extract the answer
    answer = response.choices[0].message.content.strip()

    # Extract actual source links from chunks
    # Look for "Source: ..." line in each chunk's text
    sources = []
    for chunk in chunks:
        text = chunk['text']
        # Look for "Source: ..." pattern in the text
        for line in text.split('\n'):
            if line.strip().startswith('Source:'):
                source_link = line.strip().replace('Source:', '').strip()
                if source_link:
                    sources.append(source_link)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_sources = []
    for source in sources:
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    print(f"[generate_answer] Generated answer ({len(answer)} chars)")
    print(f"[generate_answer] Sources: {len(unique_sources)} unique sources")

    return {
        'answer': answer,
        'sources': unique_sources
    }


if __name__ == "__main__":
    """
    Quick test: Retrieve and generate for a sample query.
    Run: python generate.py
    """
    from embed import retrieve

    print("=" * 80)
    print("GENERATION TEST")
    print("=" * 80)

    test_query = "What do students say about maintenance at Venue Dinkytown?"

    print(f"\nQuery: {test_query}")
    print("\n" + "-" * 80)
    print("STEP 1: Retrieval")
    print("-" * 80)

    chunks = retrieve(test_query, k=5)

    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['context']} (distance: {chunk['distance']:.4f})")

    print("\n" + "-" * 80)
    print("STEP 2: Generation")
    print("-" * 80)

    result = generate_answer(test_query, chunks)

    print("\nAnswer:")
    print(result['answer'])

    print("\nSources:")
    for source in result['sources']:
        print(f"  - {source}")

    print("\n" + "=" * 80)
