"""
Milestone 5 — Gradio Interface
Interactive Q&A interface for UMN off-campus housing reviews.

Dependencies:
  - gradio: Web UI framework

Run: python app.py
Access: http://localhost:7860
"""

import gradio as gr
from embed import retrieve
from generate import generate_answer


def answer_question(query: str) -> tuple[str, str]:
    """
    End-to-end RAG pipeline: retrieve → generate.

    Handles comparison queries by decomposing them into separate retrievals
    for each building to ensure both are represented.

    Args:
        query: User's question

    Returns:
        Tuple of (answer, sources_text)
            - answer: Generated answer string
            - sources_text: Bullet-pointed list of source files
    """
    if not query.strip():
        return "Please enter a question.", ""

    print(f"\n[RAG Pipeline] Query: '{query}'")

    # Detect comparison queries (compare, vs, versus, difference between)
    comparison_keywords = ['compare', ' vs ', ' vs.', 'versus', 'difference between', 'better']
    is_comparison = any(kw in query.lower() for kw in comparison_keywords)

    if is_comparison:
        # For comparison queries, try to extract building names and retrieve separately
        # This ensures we get chunks from both buildings
        print("[RAG Pipeline] Detected comparison query - using multi-retrieval strategy")

        # Extract potential building names (simplified heuristic)
        # List canonical names only (avoid duplicates like "Grandmarc" and "Grandmarc Seven Corners")
        buildings = [
            'Venue at Dinkytown',
            'WaHu',
            'Grandmarc Seven Corners',
            '7West',
            'Identity Dinkytown',
            'The Marshall',
            'University Commons'
        ]

        mentioned_buildings = []
        query_lower = query.lower().replace(' ', '')  # Remove spaces for flexible matching
        for building in buildings:
            # Remove spaces from building name too for matching
            building_normalized = building.lower().replace(' ', '')
            if building_normalized in query_lower:
                mentioned_buildings.append(building)

        if len(mentioned_buildings) >= 2:
            # Retrieve chunks for each building separately
            print(f"[RAG Pipeline] Found buildings: {mentioned_buildings[:2]}")
            chunks = []
            for building in mentioned_buildings[:2]:  # Compare first 2 mentioned
                building_chunks = retrieve(building, k=4)
                chunks.extend(building_chunks)
            print(f"[RAG Pipeline] Retrieved {len(chunks)} total chunks from both buildings")
        else:
            # Fallback to regular retrieval
            chunks = retrieve(query, k=8)
    else:
        # Regular query - single retrieval
        chunks = retrieve(query, k=5)

    # Step 2: Generate answer with grounding
    result = generate_answer(query, chunks)

    # Format sources as bullet list
    sources_text = "\n".join(f"• {source}" for source in result['sources'])

    return result['answer'], sources_text


# Build Gradio interface using Blocks for custom layout
with gr.Blocks(title="UMN Housing Guide") as demo:
    # Header section
    gr.Markdown(
        """
        # 🏠 Unofficial UMN Off-Campus Housing Guide
        Ask questions about student experiences at apartments near the University of Minnesota — powered by real reviews and Reddit discussions
        """
    )

    # Input section
    with gr.Row():
        query_input = gr.Textbox(
            label="Ask a question about UMN off-campus housing",
            placeholder="e.g., How is maintenance at Venue Dinkytown?",
            lines=2
        )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")

    # Output section
    with gr.Row():
        answer_output = gr.Textbox(
            label="Answer",
            lines=8,
            interactive=False
        )

    with gr.Row():
        sources_output = gr.Textbox(
            label="Sources",
            lines=4,
            interactive=False
        )

    # Wire up the button
    submit_btn.click(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )

    # Also allow Enter key to submit
    query_input.submit(
        fn=answer_question,
        inputs=query_input,
        outputs=[answer_output, sources_output]
    )

    # Example queries (optional, shown at bottom)
    gr.Examples(
        examples=[
            "What do students say about maintenance responsiveness at The Venue Dinkytown?",
            "Is Wahu considered worth the price?",
            "How do students describe safety in Dinkytown?",
            "What are the main complaints about Grandmarc Seven Corners?",
        ],
        inputs=query_input,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Launching UMN Housing Guide")
    print("=" * 80)
    print("\nAccess the interface at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("=" * 80 + "\n")

    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False  # Set to True to create a public share link
    )
