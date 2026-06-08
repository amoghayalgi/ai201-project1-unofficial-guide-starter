"""
Embeds chunks using all-MiniLM-L6-v2 and stores in ChromaDB for retrieval.

Dependencies:
  - sentence-transformers: Provides the all-MiniLM-L6-v2 embedding model
  - chromadb: Local vector store for similarity search
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# Initialize embedding model (loaded once, reused across calls)
# all-MiniLM-L6-v2: 384-dim embeddings, optimized for semantic similarity
_MODEL = None

def _get_model():
    """Lazy-load the embedding model (avoids loading on import)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL


# ChromaDB client (persistent local storage)
# Data stored in ./chroma_db/ directory
_CLIENT = chromadb.PersistentClient(path="./chroma_db")


def embed_and_store(chunks: list[dict]) -> None:
    """
    Embeds chunk text using all-MiniLM-L6-v2 and stores in ChromaDB.

    Args:
        chunks: List of chunk dicts from ingest.py. Expected format:
                Google Reviews: {'text': str, 'source': str, 'building': str, 'rating': str}
                Reddit Threads: {'text': str, 'source': str, 'context': str}

    Metadata normalization:
        - All chunks stored with 'context' field in ChromaDB
        - Google Reviews: context = building name (from 'building' key)
        - Reddit Threads: context = thread topic (from 'context' key)

    Idempotent:
        - If collection exists and has entries, skips re-embedding
        - Delete ./chroma_db/ directory to force re-embedding from scratch

    ChromaDB API notes:
        - get_or_create_collection(): Creates if missing, otherwise returns existing
        - collection.count(): Returns number of documents in collection
        - collection.add(): Stores embeddings + metadata
          - documents: List of text strings (embedded by ChromaDB if embeddings not provided)
          - metadatas: List of metadata dicts (one per document)
          - ids: List of unique string IDs (must be unique within collection)
    """
    print(f"[embed_and_store] Processing {len(chunks)} chunks...")

    # Get or create the collection
    collection = _CLIENT.get_or_create_collection(
        name="housing_reviews",
        metadata={"description": "UMN housing reviews and discussions"}
    )

    # Check if already populated (idempotency check)
    existing_count = collection.count()
    if existing_count > 0:
        print(f"[embed_and_store] Collection already has {existing_count} documents. Skipping re-embedding.")
        print(f"[embed_and_store] To re-embed from scratch, delete the ./chroma_db/ directory.")
        return

    # Prepare data for ChromaDB
    texts = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        # Text field is already formatted correctly — embed as-is
        texts.append(chunk['text'])

        # Normalize metadata: both Google and Reddit get 'context' field
        # Google: context = building name, Reddit: context = thread topic
        context = chunk.get('building') or chunk.get('context', '')

        metadatas.append({
            'source': chunk['source'],
            'chunk_index': i,
            'context': context,
        })

        # Generate unique ID (collection + chunk index)
        ids.append(f"chunk_{i}")

    print(f"[embed_and_store] Embedding {len(texts)} chunks with all-MiniLM-L6-v2...")

    # Embed all texts using the model
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Convert numpy array to list of lists for ChromaDB
    embeddings_list = embeddings.tolist()

    print(f"[embed_and_store] Storing in ChromaDB collection 'housing_reviews'...")

    # Store in ChromaDB
    # ChromaDB API: add() accepts documents + embeddings + metadatas + ids
    collection.add(
        documents=texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[embed_and_store] [OK] Stored {len(chunks)} chunks successfully.")
    print(f"[embed_and_store] Collection now has {collection.count()} documents.")


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Retrieves top-k most similar chunks for a given query.

    Args:
        query: Natural language question or search string
        k: Number of results to return (default: 5)

    Returns:
        List of dicts with keys:
            - text: The chunk text
            - source: Source file path
            - context: Building name or topic (normalized metadata)
            - chunk_index: Position in original chunk list
            - distance: Similarity distance (lower = more similar)

    ChromaDB API notes:
        - collection.query(): Performs similarity search
          - query_embeddings: List of query embedding vectors [[embedding]]
          - n_results: Number of results to return (top-k)
          - include: Which fields to include in results (defaults to all)
        - Returns dict with keys:
          - 'documents': List of lists of text strings
          - 'metadatas': List of lists of metadata dicts
          - 'distances': List of lists of distance scores
          - 'ids': List of lists of document IDs

    Distance metric:
        - ChromaDB default: L2 (Euclidean) distance
        - Lower values = more similar
        - Typical range: 0.0 (identical) to ~2.0 (very different)
    """
    print(f"[retrieve] Query: '{query}'")
    print(f"[retrieve] Retrieving top-{k} chunks...")

    # Get the collection
    collection = _CLIENT.get_collection(name="housing_reviews")

    # Embed the query using the same model
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Convert to list for ChromaDB
    query_embedding_list = query_embedding.tolist()

    # Query ChromaDB for top-k similar chunks
    # ChromaDB returns nested lists (supports batch queries), so we access [0]
    results = collection.query(
        query_embeddings=query_embedding_list,
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    # Parse results into list of dicts
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'context': results['metadatas'][0][i]['context'],
            'chunk_index': results['metadatas'][0][i]['chunk_index'],
            'distance': results['distances'][0][i],
        })

    print(f"[retrieve] [OK] Retrieved {len(chunks)} chunks.")

    return chunks

