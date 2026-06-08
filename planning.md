# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

This project focuses on student-generated knowledge about off-campus housing options near the University of Minnesota. While apartment websites provide official information about pricing and amenities, students often rely on Reddit discussions, Google reviews, and housing forums to learn about hidden issues such as maintenance responsiveness, safety concerns, noise levels, internet reliability, lease experiences, and whether apartments are worth their cost. This information is scattered across multiple platforms, making it difficult for students to make informed housing decisions.

---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->


| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Living in Dinkytown? (student discussion on safety and experiences) | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1k3x5wr/ |
| 2 | Finding Apartment Stress (housing recommendations from students) | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1nx2zvn/ |
| 3 | Identity Dinkytown – Google Reviews (exported student reviews)| Google Reviews | data/identity_google_reviews.txt |
| 4 | The Marshall – Google Reviews (exported student reviews) | Google Reviews | data/marshall_google_reviews.txt |
| 5 | Wahu Student Apartments – Google Reviews (exported student reviews)| Google Reviews | data/wahu_google_reviews.txt |
| 6 | Venue at Dinkytown – Google Reviews (exported student reviews)| Google Reviews | data/venue_google_reviews.txt |
| 7 | Grandmarc Seven Corners Recent Experience | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1sr3361/recent_experience_with_grandmarc_seven_corners/ |
| 8 | Grandmarc Seven Corners | Google Reviews | data/grandmarc_google_reviews.txt |
| 9 | University Commons Reddit | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1l9ahk5/how_actually_is_university_commons_worth_it_or_nah/ |
| 10 | University Commons | Google Reviews | data/ucommons_google_reviews.txt |
| 11 | 7West | Google Reviews | data/7west_google_reviews.txt |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**
Google Reviews - 1 review per chunk

Reddit threads - 1 comment subtree per chunk

**Overlap:**
Google Reviews: 0 overlap (independent units)
Reddit Threads: 10–20% overlap (only when splitting long threads)

**Why these choices fit your documents:**
Because your dataset is heterogeneous, each source behaves differently:

Google Reviews - sentiment analysis + ranking; must stay isolated (no overlap needed)
Reddit threads - context-dependent opinions; needs partial overlap so meaning isn’t lost between replies

**Final chunk count:** 126 (70 google reviews and 56 reddit comments across all the sources)

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 via sentence-transformers

**Top-k:** 5

**Production tradeoff reflection:** For this project, all-MiniLM-L6-v2 is a strong fit — it runs locally with no API key, handles short opinion-style text well, and has a 256-token context window that matches the size of most reviews and Reddit comments in this corpus. In a real production deployment, I'd weigh several tradeoffs. A larger model like text-embedding-3-large (OpenAI) would improve accuracy on nuanced queries but adds API cost and latency per query. If users were international students, multilingual support (e.g., paraphrase-multilingual-MiniLM-L12-v2) would matter. For a high-traffic system, I'd trade some accuracy for a smaller, faster model like all-MiniLM-L6-v2 with caching. Local models eliminate data privacy concerns (student reviews going to a third-party API), which is relevant for a university platform

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about maintenance responsiveness at The Venue Dinkytown? | Reviews and Reddit threads indicate maintenance is slow to respond, with multiple students citing weeks-long wait times for basic repairs. |
| 2 | Is Wahu considered worth the price by students? | Mixed — some students find the amenities justify the cost, but several reviews criticize the price-to-quality ratio, especially given reported noise and management issues. |
| 3 | What are the main complaints students have about Grandmarc Seven Corners? | Common complaints include management unresponsiveness, issues with package delivery/security, and concerns raised in a recent Reddit thread about lease experience. |
| 4 | How do students describe the safety situation in Dinkytown? | Students on the r/uofmn Dinkytown thread describe mixed safety perceptions — some cite concerns about nighttime safety, others note it has improved. No consensus that it is outright dangerous. |
| 5 | Which apartment building gets the most consistently positive reviews from students? | Based on Google Reviews and Reddit, The Marshall tends to receive more positive mentions around management and quality, though individual experiences vary. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Sarcasm and vague sentiment in reviews. Google Reviews frequently contain ambiguous language ("great for the price 🙃") or sarcasm that the embedding model will likely misinterpret as positive. A query like "is Wahu worth it?" could retrieve a sarcastically positive review and pass it to the LLM as genuine praise. This is a retrieval quality problem — the semantic embedding won't distinguish ironic from sincere text, meaning the generation step may produce a misleadingly rosy answer.
2. Information scattered across chunk boundaries. Reddit comment subtrees sometimes split key context: a student's complaint about mold appears in one comment, but the follow-up confirming it's a widespread issue is in a reply two levels down. If those end up in separate chunks with insufficient overlap, neither chunk alone will be retrievable for a query like "do any Dinkytown apartments have mold problems?" — the relevant signal is split, and retrieval will miss it entirely.
---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->
Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
(Python/requests)  (LangChain  (all-MiniLM-L6-v2 /       (ChromaDB   (Groq API /
                   TextSplitter) ChromaDB)                  top-k=5)   llama-3.3-70b)
                                                                 ↑
                                                         Gradio web UI (user query)
---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
I will give Claude my Document Sources table (listing all 13 sources and their types) and my Chunking Strategy section, and ask it to implement an ingest.py script with two functions: load_google_reviews(filepath) that reads .txt review files into individual strings, and load_reddit_thread(url_or_file) that parses comment subtrees. I'll also ask it to implement chunk_text(docs, strategy) using my specified chunk size and overlap. I'll verify the output by printing 5 random chunks and checking that each is self-contained and matches my chunking spec — if any chunk contains HTML artifacts or is shorter than one complete thought, I'll debug before moving on.

**Milestone 4 — Embedding and retrieval:**
I will give Claude my Retrieval Approach section and the pipeline diagram, and ask it to implement embed.py that loads chunks from the ingestion pipeline, embeds them with all-MiniLM-L6-v2, and stores them in ChromaDB with source metadata (filename, chunk index). I'll also ask it to implement a retrieve(query, k=5) function. I'll verify by running 3 of my evaluation plan queries and checking that returned chunks visibly relate to each question and have distance scores below 0.5.

**Milestone 5 — Generation and interface:**
I will give Claude my grounding requirement (answers from retrieved chunks only, with explicit source attribution), the expected output format (answer + bulleted source list), and the Gradio skeleton structure from the project spec. I'll ask it to wire together retrieve() and a Groq API call with a system prompt that enforces grounding. Before running the code, I'll read the system prompt it generates and verify it explicitly instructs the model to refuse answering if the retrieved context doesn't contain the answer — not just "use the context" but a hard fallback instruction.

