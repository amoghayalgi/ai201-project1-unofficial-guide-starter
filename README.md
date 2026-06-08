# The Unofficial Guide — Project 1

---

## Domain

This project focuses on student-generated knowledge about off-campus housing options near the University of Minnesota. While apartment websites provide official information about pricing and amenities, students often rely on Reddit discussions, Google reviews, and housing forums to learn about hidden issues such as maintenance responsiveness, safety concerns, noise levels, internet reliability, lease experiences, and whether apartments are worth their cost. This information is scattered across multiple platforms, making it difficult for students to make informed housing decisions.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Living in Dinkytown? (student discussion on safety and experiences) | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1k3x5wr/ |
| 2 | Finding Apartment Stress (housing recommendations from students) | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1nx2zvn/ |
| 3 | Grandmarc Seven Corners Recent Experience | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1sr3361/ |
| 4 | University Commons Reddit (worth it or not discussion) | Reddit Thread | https://www.reddit.com/r/uofmn/comments/1l9ahk5/ |
| 5 | Identity Dinkytown – Google Reviews (exported student reviews)| Google Reviews | data/identity_google_reviews.txt |
| 6 | The Marshall – Google Reviews (exported student reviews) | Google Reviews | data/marshall_google_reviews.txt |
| 7 | Wahu Student Apartments – Google Reviews (exported student reviews)| Google Reviews | data/wahu_google_reviews.txt |
| 8 | Venue at Dinkytown – Google Reviews (exported student reviews)| Google Reviews | data/venue_google_reviews.txt |
| 9 | Grandmarc Seven Corners – Google Reviews | Google Reviews | data/grandmarc_google_reviews.txt |
| 10 | University Commons – Google Reviews | Google Reviews | data/ucommons_google_reviews.txt |
| 11 | 7West Apartments – Google Reviews | Google Reviews | data/7west_google_reviews.txt |

**Total:** 11 sources (7 Google Review files + 4 Reddit threads)

---

## Chunking Strategy

**Chunk size:**
- **Google Reviews:** 1 review per chunk (up to 2,000 characters; longer reviews split with 200-char overlap)
- **Reddit threads:** 1 comment subtree per chunk (up to 800 characters; longer subtrees split with 120-char overlap ~15%)

**Overlap:**
- **Google Reviews:** 10% overlap (200 chars) when splitting long reviews — preserves context at boundaries while avoiding complete duplication
- **Reddit Threads:** 15% overlap (120 chars) when splitting long comment chains — maintains conversational context across split points

**Why these choices fit the documents:**

The dataset is heterogeneous with two distinct source types requiring different strategies:

1. **Google Reviews** are independent opinion units with no narrative dependency. Each review expresses a complete sentiment about one building. Minimal overlap (10%) is sufficient for the rare long review that exceeds 2,000 characters. Most reviews stay intact as single chunks.

2. **Reddit threads** contain nested conversations where meaning flows across comments and replies. A 15% overlap ensures that when a long comment chain is split (e.g., a detailed complaint followed by multiple confirming replies), the split chunks share enough context to remain retrievable together. The 800-character threshold matches the typical length of a comment with 2-3 replies.

**Preprocessing:**
- Stripped "Review N" and "Comment N" headers (no semantic value)
- Prepended building name and rating to each Google Review chunk
- Prepended discussion topic to each Reddit chunk
- Added source line (URL or name) to each chunk for attribution

**Final chunk count:** 125 chunks (75 Google Reviews + 50 Reddit comments)

---

## Sample Chunks

### Chunk 1
**Source:** Venue Google Reviews  
**Building:** Venue at Dinkytown  
**Rating:** 1/5  
**Length:** 263 characters

```
Source: Venue Google Reviews
Building: Venue at Dinkytown
Rating: 1/5
Severe maintenance and management issues. Washer, dryer, and fridge were broken for over a month. Took 6 weeks to fix appliances. Promised compensation was never delivered. Management is inconsistent and unhelpful.
```

### Chunk 2
**Source:** https://www.reddit.com/r/uofmn/comments/1k3x5wr/  
**Context:** Dinkytown Safety  
**Length:** 227 characters

```
Source: https://www.reddit.com/r/uofmn/comments/1k3x5wr/living_in_dinkytown/
Discussion: Dinkytown Safety
That's where most students live and nearly all of them are just fine. Always people around. Exercise common sense like you would in any major city.
```

### Chunk 3
**Source:** Wahu Google Reviews  
**Building:** WaHu Student Apartments  
**Rating:** 1/5  
**Length:** 238 characters

```
Source: Wahu Google Reviews
Building: WaHu Student Apartments
Rating: 1/5
Management is awful. Many 5-star reviews are believed to be incentivized. Amenities are okay but not worth the problems. Room and facilities are acceptable but management is the biggest issue.
```

### Chunk 4
**Source:** University Commons Google Reviews  
**Building:** University Commons  
**Rating:** 2/5  
**Length:** 405 characters

```
Source: University Commons Google Reviews
Building: University Commons
Rating: 2/5
Place is average overall. Move-in condition was very dirty with dead flies, dust, and hair in bathroom and carpet. Maintenance request for cleaning was ignored. Location is excellent. Gym and pool are nice amenities. Study spaces are too small and noisy. Management is rude and unhelpful, but leasing staff is kind. Parking is overpriced.
```

### Chunk 5
**Source:** https://www.reddit.com/r/uofmn/comments/1k3x5wr/  
**Context:** Dinkytown Safety  
**Length:** 397 characters

```
Source: https://www.reddit.com/r/uofmn/comments/1k3x5wr/living_in_dinkytown/
Discussion: Dinkytown Safety
Makes me sad to hear people say "don't wander around alone at night." I was a student from 2015-2018 and never really had to worry about that walking from the bars or to another party.

- Reply 1
You not worrying and there not being a risk are separate things
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via sentence-transformers

**Why this model:** 
- Runs locally with no API key required
- 384-dimensional embeddings optimized for semantic similarity
- 256-token context window fits most reviews and Reddit comments
- Fast inference (~50ms per query on CPU)
- Well-suited for short opinion-style text

**Production tradeoff reflection:**

For this project, all-MiniLM-L6-v2 is a strong fit — it runs locally with no API key, handles short opinion-style text well, and has a 256-token context window that matches the size of most reviews and Reddit comments in this corpus.

In a real production deployment, I'd weigh several tradeoffs:

1. **Accuracy vs. Cost:** A larger model like `text-embedding-3-large` (OpenAI) would improve accuracy on nuanced queries (e.g., distinguishing sarcasm, handling synonyms better) but adds API cost (~$0.13 per 1M tokens) and latency (200-300ms per query). For a high-traffic student platform, this could exceed budget.

2. **Multilingual Support:** If the user base included international students posting reviews in other languages, a multilingual model like `paraphrase-multilingual-MiniLM-L12-v2` would be essential. The current model only handles English well.

3. **Domain Adaptation:** Fine-tuning on housing review data could improve retrieval for domain-specific terms (e.g., "utilities included" vs. "utility bills"). This requires labeled training data and GPU resources.

4. **Privacy:** Local models eliminate data privacy concerns — student reviews never leave the server. Using an API-hosted model (OpenAI, Cohere) means sensitive housing complaints go to a third party, which might violate university data policies.

5. **Latency:** For real-time chat, a smaller model like `all-MiniLM-L6-v2` (20MB) loads in <1 second. Larger models (2-5GB) add startup latency, problematic for serverless deployments.

**Choice for this project:** `all-MiniLM-L6-v2` balances speed, privacy, and quality for the student housing use case.

---

## Retrieval Test Results

### Query 1: "What do students say about maintenance responsiveness at The Venue Dinkytown?"

**Top 5 Retrieved Chunks:**

| Rank | Context | Source | Distance | Preview |
|------|---------|--------|----------|---------|
| 1 | Venue at Dinkytown | Venue Google Reviews | 0.7496 | "Maintenance responds quickly. Hallways and trash rooms are well kept..." (Rating: 4/5) |
| 2 | Venue at Dinkytown | Venue Google Reviews | 0.7505 | "Washer, dryer, and fridge were broken for over a month. Took 6 weeks to fix..." (Rating: 1/5) |
| 3 | Venue at Dinkytown | Venue Google Reviews | 0.8083 | "Lock on unit took months to fix. Maintenance was very slow and unreliable..." (Rating: 1/5) |
| 4 | Venue at Dinkytown | Venue Google Reviews | 0.8265 | "Issues are fixed quickly, often same or next day..." (Rating: 5/5) |
| 5 | Venue at Dinkytown | Venue Google Reviews | 0.8322 | "Maintenance is fast and responsive..." (Rating: 5/5) |

**Why these chunks are relevant:**
All 5 chunks are from Venue at Dinkytown and directly address maintenance responsiveness. The retrieval captured both positive experiences (ranks 1, 4, 5: "responds quickly", "same or next day") and negative experiences (ranks 2, 3: "6 weeks to fix", "months to fix"), providing balanced evidence for the query. Distance scores (0.75-0.83) indicate moderate semantic similarity — higher than ideal but acceptable for opinion text where phrasing varies widely. The metadata prepending strategy successfully filtered to only Venue chunks.

---

### Query 2: "Is Wahu considered worth the price?"

**Top 5 Retrieved Chunks:**

| Rank | Context | Source | Distance | Preview |
|------|---------|--------|----------|---------|
| 1 | WaHu Student Apartments | Wahu Google Reviews | 0.7812 | "Apartment is good value for price..." (Rating: 4/5) |
| 2 | WaHu Student Apartments | Wahu Google Reviews | 0.8149 | "Premium view fees are misleading. Management is terrible..." (Rating: 1/5) |
| 3 | WaHu Student Apartments | Wahu Google Reviews | 0.8555 | "Tour experience was great. Beautiful building..." (Rating: 5/5) |
| 4 | WaHu Student Apartments | Wahu Google Reviews | 0.8891 | "Amenities are okay but not worth the problems..." (Rating: 1/5) |
| 5 | WaHu Student Apartments | Wahu Google Reviews | 0.8923 | "Parking is overpriced. Management is rude..." (Rating: 2.5/5) |

**Why these chunks are relevant:**
All 5 chunks discuss price/value at WaHu. The retrieval found a mix of ratings (4/5, 1/5, 5/5, 1/5, 2.5/5), providing evidence for both sides of the price question. Chunk 1 explicitly states "good value for price," while chunks 2, 4, and 5 criticize fees, management issues, and overpricing. Distance scores are higher (0.78-0.89), likely because "worth the price" is an abstract concept with varied phrasing in reviews (e.g., "value", "cost", "fees", "overpriced"). The semantic search correctly identified these as related despite different wording.

---

### Query 3: "How do students describe safety in Dinkytown?"

**Top 5 Retrieved Chunks:**

| Rank | Context | Source | Distance | Preview |
|------|---------|--------|----------|---------|
| 1 | Dinkytown Safety | Reddit Thread | 0.3768 | "Most students are just fine. Exercise common sense like you would in any major city..." |
| 2 | Venue at Dinkytown | Venue Google Reviews | 0.5679 | "Safety was the top concern. Property exceeded expectations in safety..." (Rating: 5/5) |
| 3 | Dinkytown Safety | Reddit Thread | 0.6298 | "It's only bad in the summer" |
| 4 | Dinkytown Safety | Reddit Thread | 0.6393 | "It's actually much safer than the area around dinky because there are so many people around" |
| 5 | Dinkytown Safety | Reddit Thread | 0.6526 | "Don't wander around alone at night... I never really had to worry about that..." |

**Why these chunks are relevant:**
This is the **strongest retrieval** of the three queries. The top result has a distance of 0.38 — well below the 0.5 threshold for high-quality matches. All 5 chunks directly discuss Dinkytown safety (4 from the Reddit "Dinkytown Safety" thread, 1 from Venue reviews). The retrieval captured multiple perspectives: general safety with precautions (rank 1), seasonal variation (rank 3), comparative safety (rank 4), and nighttime concerns (rank 5). The low distance scores indicate the embedding model strongly associated "safety in Dinkytown" with these discussion contexts.

---

## Grounded Generation

**System prompt grounding instruction:**

```
You are a helpful assistant that answers questions about off-campus housing 
near the University of Minnesota based on student reviews and discussions.

GROUNDING RULES:
1. Answer ONLY using the provided context chunks below. Do not use any 
   external knowledge.

2. Synthesize information across multiple chunks to provide comprehensive 
   answers. This includes:
   - Aggregating feedback about a single building
   - Comparing different buildings when asked
   - Identifying patterns across multiple reviews

3. Do NOT cite sources in your answer text. This includes:
   - No phrases like "According to...", "Students mention...", "Reviews indicate..."
   - No reference numbers like [1], [2], [3] or (Source 1), etc.
   - Just provide the information directly
   The sources will be shown separately to the user.

4. For comparison questions (e.g., "compare X and Y"), provide a structured 
   comparison even if the chunks don't explicitly compare them side-by-side.

5. If the question is completely unrelated to the retrieved context (e.g., 
   asking about restaurants when all chunks discuss housing), respond with 
   EXACTLY: "I don't have enough information in my sources to answer that question."

6. Do NOT refuse just because:
   - The answer requires synthesis across chunks
   - You need to compare buildings that aren't mentioned together in a single chunk
   - The evidence is mixed or contradictory

7. Be direct and helpful. Acknowledge when evidence is mixed (e.g., "Feedback 
   is mixed — some residents report fast maintenance while others experienced 
   delays").
```

**How source attribution is surfaced in the response:**

Source attribution is separated from the answer text to maintain readability:

1. **Answer box:** Clean, citation-free response synthesizing information from retrieved chunks
2. **Sources box:** Bullet-pointed list showing:
   - For Google Reviews: Source name (e.g., "Venue Google Reviews")
   - For Reddit: Full URL (e.g., "https://www.reddit.com/r/uofmn/comments/1k3x5wr/")

Each chunk has a "Source:" line prepended during ingestion. When generating an answer, the system:
1. Passes chunks with source lines to the LLM
2. Extracts unique source names/URLs from the chunks used
3. Displays them in the dedicated Sources box

This approach prevents citation clutter in the answer while ensuring full traceability.

---

## Example Responses

### Example 1: In-Scope Query with Attribution

**Query:** "What do students say about maintenance responsiveness at The Venue Dinkytown?"

**Answer:**
> Feedback is mixed about maintenance responsiveness at The Venue Dinkytown. Some residents report that maintenance responds quickly, while others experienced severe delays, with issues taking months to resolve. One resident mentioned that appliances were broken for over a month and it took 6 weeks to fix them, while another resident had a lock that took months to fix. In contrast, one review stated that maintenance responds quickly, indicating inconsistent maintenance responsiveness.

**Sources:**
- Venue Google Reviews

---

### Example 2: Comparison Query with Attribution

**Query:** "Compare Grandmarc Seven Corners and 7West apartments"

**Answer:**
> Comparing Grandmarc Seven Corners and 7 West Apartments, there are notable differences in resident experiences. Grandmarc Seven Corners has mixed reviews, with some residents having a great experience and planning to return, while others report significant security concerns, such as car break-ins in the gated parking garage, and poor management response to issues like WiFi outages. Some residents find the building to be average, with older infrastructure but decent maintenance and amenities like the gym and courtyard.
>
> In contrast, 7 West Apartments consistently receives positive reviews, with residents praising the helpful and friendly leasing office and maintenance staff, excellent management team, and reliable maintenance services. The location near campus and downtown is also a plus. Residents appreciate the clean and quiet units, modern appliances, and community events that add value to their living experience. Overall, 7 West Apartments seems to offer a more consistently positive experience, with responsive management and a strong sense of community, whereas Grandmarc Seven Corners has more variable reviews and significant concerns about safety and management responsiveness.

**Sources:**
- Grandmarc Seven Corners Google Reviews
- 7 West Apartments Google Reviews

---

### Example 3: Out-of-Scope Query Showing Refusal

**Query:** "What is the monthly rent for a 2-bedroom apartment at Venue Dinkytown?"

**Answer:**
> I don't have enough information in my sources to answer that question. The context provided discusses various aspects of living at Venue at Dinkytown, including reviews and experiences, but does not mention specific monthly rent prices for a 2-bedroom apartment.

**Sources:**
- Venue Google Reviews

**Analysis:** The system correctly refused because the retrieved chunks (distance range 0.73-0.86, higher than typical) contained general reviews about Venue but no specific pricing information. The grounding mechanism prevented hallucination and triggered the refusal response.

---

## Query Interface

**Interface:** Gradio web UI accessible at `http://localhost:7860`

**Input fields:**
- **Question input box:** Multi-line text field labeled "Ask a question about UMN off-campus housing"
  - Accepts natural language queries
  - Placeholder: "e.g., How is maintenance at Venue Dinkytown?"
  - Supports comparison queries, building-specific questions, and general housing topics

**Output fields:**
- **Answer box:** 8-line text area labeled "Answer"
  - Displays the generated response without citations
  - Non-editable
- **Sources box:** 4-line text area labeled "Sources"
  - Shows bullet-pointed list of source names/URLs
  - One unique source per line
  - Non-editable

**Sample interaction transcript:**

```
User Input:
┌─────────────────────────────────────────────────────────────┐
│ How do students describe safety in Dinkytown?              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        [Ask]

System Output:

Answer:
┌─────────────────────────────────────────────────────────────┐
│ Students generally describe Dinkytown as a safe area,       │
│ especially when exercising common sense and being aware of  │
│ their surroundings, similar to any major city. Many         │
│ students feel safe walking alone in the area, with one      │
│ student mentioning they felt safer in Dinkytown than on     │
│ campus. The presence of a lot of people around and          │
│ increased police presence in the area contribute to the     │
│ sense of security. However, it's noted that the farther     │
│ you go from campus, the area can become shadier, but        │
│ overall, most students have no issues and feel fine living  │
│ in Dinkytown.                                                │
└─────────────────────────────────────────────────────────────┘

Sources:
┌─────────────────────────────────────────────────────────────┐
│ • https://www.reddit.com/r/uofmn/comments/1k3x5wr/          │
│   living_in_dinkytown/                                       │
│ • Venue Google Reviews                                       │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- Enter key submits the query (in addition to clicking "Ask")
- Example queries shown below the interface for quick testing
- Comparison query support (detects keywords: "compare", "vs", "better")
- Handles both building-specific and general housing questions

---

## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do students say about maintenance responsiveness at The Venue Dinkytown? | Feedback is mixed, but it leans toward generally fast and responsive maintenance | Feedback is mixed — some report quick response, others cite severe delays (6 weeks, months to fix) | Relevant (5/5 Venue chunks, distances 0.75-0.83) | **Accurate** |
| 2 | Is Wahu considered worth the price? | Most residents do not feel WaHu is worth the price. Amenities valued by some, but price-to-quality ratio criticized | Feedback is mixed — one finds it good value, others express concerns about fees, management, and overpricing | Relevant (5/5 WaHu chunks, distances 0.78-0.89) | **Partially Accurate** |
| 3 | How do students describe safety in Dinkytown? | Students generally feel Dinkytown is safe enough to live in, as long as you use basic city awareness, especially at night | Generally safe with common sense precautions, presence of people and police contribute to security, but farther from campus can be shadier | Relevant (4/5 Dinkytown Safety, distances 0.38-0.65) | **Accurate** |
| 4 | What are the main complaints about Grandmarc Seven Corners? | Management unresponsiveness, package security, lease experience issues, WiFi problems | Unhelpful staff, poorly maintained rooms, outdated apartments, WiFi outages, poor communication, unexplained fees after move-out | Relevant (5/5 Grandmarc chunks, distances 0.51-0.59) | **Accurate** |
| 5 | What is the monthly rent for a 2-bedroom apartment at Venue Dinkytown? | Should refuse or give inaccurate answer - rent prices not in review corpus | Correctly refused: "I don't have enough information in my sources to answer that question." | Partially relevant (4/5 Venue chunks but no price info, distances 0.73-0.86) | **N/A (Correct Refusal)** |

**Key observations:**
- **Query 3 (Dinkytown safety)** had the strongest retrieval (distance 0.38 on top result)
- **Queries 1, 2, 4** had moderate distance scores (0.75-0.89) but still retrieved topically correct chunks
- **Query 5** correctly triggered the refusal mechanism when source information was unavailable
- All queries successfully filtered to the relevant buildings/topics via metadata prepending strategy

---

## Failure Case Analysis

**Question that failed:**

"Is Wahu considered worth the price?"

**What the system returned:**

> "Feedback about WaHu Student Apartments being worth the price is mixed. One resident finds the apartment to be good value for the price, appreciating the fast maintenance and helpful staff. However, others have expressed concerns about the cost, with one resident feeling that the 'premium view' fees are misleading and another stating that the amenities are not worth the problems they've experienced."

**Accuracy judgment:** Partially Accurate

**Expected answer:** "**Most** residents do not feel WaHu is worth the price."  
**Actual answer:** "Feedback is **mixed**."

**Root cause (tied to a specific pipeline stage):**

The failure occurred at the **generation stage** due to two interacting factors:

1. **Retrieval returned a rating-diverse set of chunks** (1/5, 4/5, 5/5, 1/5, 2.5/5) with distance scores of 0.78-0.89. While all chunks discuss price/value at WaHu, the semantic similarity scoring doesn't weight by rating — a 5-star "good value" review has similar embedding distance to a 1-star "not worth it" review because both contain the words "price" and "value."

2. **The LLM treated all chunks as equally weighted** during synthesis. The system prompt instructs the model to "synthesize information across multiple chunks" and "acknowledge when evidence is mixed," which is correct behavior. However, it doesn't instruct the model to perform sentiment aggregation or count the ratio of positive vs. negative reviews. As a result, the model saw {1 positive, 3 negative, 1 neutral} and concluded "mixed" rather than recognizing the **predominance** of negative sentiment.

The issue is **not** a retrieval failure (all chunks are relevant) or a refusal failure (the model attempted to answer). It's a **synthesis calibration** problem: the model balanced conflicting evidence rather than quantifying the distribution.

**What you would change to fix it:**

1. **Add rating-aware context formatting:** Instead of passing chunks as plain text, format each with an explicit rating indicator:
   ```
   [★☆☆☆☆ - 1/5] "Not worth the problems..."
   [★★★★☆ - 4/5] "Good value for price..."
   ```
   This makes the rating distribution visually salient to the LLM.

2. **Update the system prompt with sentiment aggregation guidance:**
   ```
   When synthesizing reviews with ratings, note the distribution:
   - If 4+ out of 5 chunks have low ratings (1-2 stars), state "most residents" 
     feel negatively
   - If ratings are evenly split (2-3 positive, 2-3 negative), state "mixed"
   ```

3. **Pre-filter chunks by rating** for price-value queries: Retrieve top-10, then filter to the 5 most relevant by a weighted score: `relevance_score = (1 - distance) * rating_alignment`, where rating_alignment favors majority sentiment. This would bias toward the predominant viewpoint while still showing counterevidence.

**Alternative explanation (architectural):** This could also be framed as a **retrieval diversity vs. consensus** tradeoff. The current top-k=5 strategy prioritizes semantic similarity, which returns a representative sample of all price-related reviews. A consensus-focused approach would retrieve top-15 and filter to the 5 that agree, sacrificing balanced evidence for clearer directional answers. The choice depends on whether the use case prioritizes nuanced truth (current) or actionable recommendations (alternative).

---

## Spec Reflection

**One way the spec helped you during implementation:**

The **chunking strategy section** in planning.md prevented a common RAG mistake: using a fixed chunk size across heterogeneous documents. By forcing me to think through chunk sizes and overlap *before* coding, I recognized that Google Reviews (independent units) and Reddit threads (nested conversations) have fundamentally different information structures. This led to the decision to chunk reviews 1-per-chunk with minimal overlap, while giving Reddit threads overlap to preserve conversational context. Without pre-planning, I would likely have used LangChain's default 500-character fixed split, which would have broken mid-review and mid-reply-chain, degrading retrieval quality. The spec's requirement to justify chunk sizes with reasoning ("why it fits your documents") made this structural difference explicit early, shaping the entire ingestion pipeline.

**One way your implementation diverged from the spec, and why:**

The original spec anticipated using Reddit's API (via PRAW) to scrape threads live, with environment variables for API credentials. In practice, I diverged to **using local .txt files** for Reddit data instead. This happened because during Milestone 3, I discovered that Reddit's API rate limits (60 requests/min) and authentication complexity would slow down development and testing — every re-run of the ingestion script would hit the API. More importantly, storing threads as .txt files made the dataset **reproducible**: the same chunks would be generated every time, essential for comparing retrieval results across iterations. The tradeoff is that the data is now static (won't pick up new Reddit comments), but for an academic project with evaluation requirements, reproducibility outweighed freshness. The spec's PRAW integration code remains in `ingest.py` as a `_load_reddit_url()` function but is unused; all threads are loaded via `_load_reddit_file()` instead.

---

## AI Usage

**Instance 1: Document Ingestion Pipeline (Milestone 3)**

- **What I gave the AI:**
  - The Document Sources table from planning.md (listing 11 sources with file paths and URLs)
  - The Chunking Strategy section (specifying 1 review/chunk for Google, 1 comment subtree/chunk for Reddit, with 10-20% overlap rules)
  - A sample of my data files showing the format (Google reviews separated by `---`, Reddit comments with reply threads)
  - Instructions to implement `load_google_reviews()`, `load_reddit_thread()`, and `chunk_text()` functions

- **What it produced:**
  - An `ingest.py` script with three functions using LangChain's RecursiveCharacterTextSplitter
  - Google review parser that split on `---` separators and returned one chunk per review
  - Reddit parser that supported both URL (via PRAW) and local file loading
  - Chunking function with separate strategies for Google (no overlap) and Reddit (15% overlap)

- **What I changed or overrode:**
  - **Removed the Reddit URL loading functionality** — changed to only use local .txt files (see Spec Reflection for reasoning)
  - **Added metadata prepending** — the AI initially returned chunks with just the review/comment text. I modified `load_google_reviews()` to prepend `"Source: <name>\nBuilding: <building>\nRating: <rating>\n"` and `load_reddit_thread()` to prepend `"Source: <URL>\nDiscussion: <topic>\n"`. This was critical for the embedding model to associate chunks with their building/topic context.
  - **Fixed Google review header parsing** — the AI's parser included the file header (Apartment name, Source line) in the first chunk's text, creating duplicate metadata. I added a skip check: `if not re.search(r"^Review\s+\d+", block, re.MULTILINE): continue` to filter out the header block before processing reviews.

**Instance 2: Grounded Generation System Prompt (Milestone 5)**

- **What I gave the AI:**
  - My grounding requirement: "Answers must come only from retrieved chunks, with explicit source attribution, and refuse when chunks don't contain relevant information"
  - The evaluation plan queries, including one out-of-scope query ("What is the best restaurant near campus?")
  - The chunk format showing source metadata
  - My concern about over-refusal: "The system should allow synthesis across chunks and not refuse just because the answer requires light inference"

- **What it produced:**
  - A system prompt with 6 grounding rules emphasizing strict adherence to context
  - Instructions to cite sources using phrases like "According to Venue at Dinkytown reviews..." and reference numbers [1], [2]
  - A refusal instruction: respond with "I don't have enough information..." when context is off-topic
  - Default behavior to synthesize across chunks

- **What I changed or overrode:**
  - **Removed citation requirements from the answer text** — the AI's prompt told the model to cite sources inline, but I separated attribution into a dedicated Sources box for cleaner readability. I added Rule 3: "Do NOT cite sources in your answer text. This includes no phrases like 'According to...', no reference numbers like [1], [2]."
  - **Added explicit comparison query handling** — the AI's prompt didn't address comparison queries ("Compare X and Y"). I added Rule 4 to explicitly permit structured comparisons even when buildings aren't mentioned together in a single chunk.
  - **Calibrated the refusal criteria** — the AI's prompt said "refuse if context doesn't contain the answer," which was too vague. I made it more specific: "If the question is **completely unrelated** to the retrieved context (e.g., asking about restaurants when all chunks discuss housing), respond with EXACTLY: [refusal phrase]." This reduced false refusals while maintaining grounding.
  - **Added acknowledgment of mixed evidence** — I inserted Rule 7 to explicitly encourage phrases like "Feedback is mixed — some report X while others mention Y," which the AI's original prompt didn't emphasize. This improved answer quality for queries with genuinely split opinion (e.g., Venue maintenance).

**Overall pattern:** The AI produced structurally correct code and prompts, but I consistently had to refine the **interface between components** (metadata format, citation separation) and **calibrate thresholds** (refusal criteria, synthesis vs. refusal balance) to match my specific use case. The AI excelled at implementing the "what" (functions, prompts) but required human judgment for the "how much" (overlap percentages, chunk size limits, grounding strictness).

---

## Final Notes

**Total implementation time:** ~10-12 hours across 6 milestones

**Final system stats:**
- 11 document sources (7 Google Review files, 4 Reddit threads)
- 125 chunks embedded in ChromaDB
- all-MiniLM-L6-v2 embedding model (384 dimensions)
- Groq llama-3.3-70b-versatile for generation
- Gradio web interface at localhost:7860

**What works well:**
- Building-specific queries (retrieval filters correctly via metadata)
- Safety/general topic queries (strong semantic matches)
- Comparison queries (query decomposition retrieves from both buildings)
- Out-of-scope refusal (correctly identifies when context lacks information)

**Known limitations:**
- Doesn't weight by star rating (treats all reviews equally during synthesis)
- No temporal awareness (can't answer "has X improved recently?")
- Limited to UMN housing (transfer learning to other universities would require re-embedding)
- Sarcasm detection (embedding model treats ironic reviews as sincere)

## Project Structure
```text
.
├── data/                                    # Source documents
│   ├── identity_google_reviews.txt
│   ├── marshall_google_reviews.txt
│   ├── wahu_google_reviews.txt
│   ├── venue_google_reviews.txt
│   ├── grandmarc_google_reviews.txt
│   ├── ucommons_google_reviews.txt
│   ├── 7west_google_reviews.txt
│   ├── living_in_dinkytown_reddit.txt
│   ├── finding_apartment_uofm_reddit.txt
│   ├── grandmarc_seven_corners_reddit.txt
│   └── university_commons_reddit.txt
│
├── chroma_db/                               # ChromaDB vector database (generated)
│   └── [vector embeddings stored here]
│
├── ingest.py                                # Milestone 3: Document loading and chunking
├── embed.py                                 # Milestone 4: Embedding generation and retrieval
├── generate.py                              # Milestone 5: Grounded response generation using Groq
├── app.py                                   # Milestone 5: Gradio web interface
│
├── test_retrieval.py                        # Retrieval evaluation (3 test queries)
├── test_generation.py                       # End-to-end generation tests
├── run_evaluation.py                        # Full evaluation suite (5 test queries)
├── verify_ingestion.py                      # Ingestion verification helper
│
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment variables template
├── .gitignore                               # Git ignore rules
│
├── planning.md                              # Pre-implementation planning document
└── README.md                                # Project documentation
