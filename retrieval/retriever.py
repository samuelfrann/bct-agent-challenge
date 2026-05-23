import json
import chromadb

from sentence_transformers import SentenceTransformer
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHROMA_DIR = str(ROOT / "chroma_db")

_client       = None
_embedder     = None
_personas_col = None
_reviews_col  = None
_business_col = None

def _init():
    global _client, _embedder, _personas_col, _reviews_col, _business_col
    if _client is None:
        _embedder     = SentenceTransformer("all-MiniLM-L6-v2")
        _client       = chromadb.PersistentClient(path=CHROMA_DIR)
        _personas_col = _client.get_collection("user_personas")
        _reviews_col  = _client.get_collection("user_reviews")
        _business_col = _client.get_collection("businesses")

        # ── 1. Fetch one user's persona by ID ─────────────────────────────────────────
def get_persona(user_id: str) -> dict | None:
    """Returns full persona dict or None if user not found."""
    _init()
    try:
        result = _personas_col.get(
            ids=[user_id],
            include=["metadatas", "documents"]
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {
            "user_id":           meta["user_id"],
            "avg_rating":        meta["avg_rating"],
            "rating_variance":   meta["rating_variance"],
            "avg_review_length": meta["avg_review_length"],
            "total_reviews":     meta["total_reviews"],
            "tone_tags":         json.loads(meta["tone_tags"]),
            "top_categories":    json.loads(meta["top_categories"]),
            "recent_reviews":    json.loads(meta["recent_reviews"]),
            "summary":           result["documents"][0]
        }
    except Exception:
        return None

# ── 2. Find users who behave like a given description (cold-start) ────────────
def find_similar_users(description: str, k: int = 5) -> list[dict]:
    """
    Embed a plain-text user description and return k most similar personas.
    Used when a new user has no history — borrow signal from behavioral twins.
    """
    _init()
    embedding = _embedder.encode(description).tolist()
    results   = _personas_col.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )
    users = []
    for i, meta in enumerate(results["metadatas"][0]):
        users.append({
            "user_id":          meta["user_id"],
            "avg_rating":       meta["avg_rating"],
            "tone_tags":        json.loads(meta["tone_tags"]),
            "top_categories":   json.loads(meta["top_categories"]),
            "recent_reviews":   json.loads(meta["recent_reviews"]),
            "summary":          results["documents"][0][i],
            "similarity_score": round(1 - results["distances"][0][i], 4)
        })
    return users

# ── 3. Get a user's stored reviews, optionally filtered by topic ──────────────
def get_user_reviews(user_id: str, query: str = None, k: int = 5) -> list[dict]:
    """
    If query is provided: returns k reviews semantically closest to the query.
    If no query: returns the user's most recent k reviews.
    Used to show the LLM real examples of how this person writes.
    """
    _init()

    if query:
        embedding = _embedder.encode(query).tolist()
        results   = _reviews_col.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        return [
            {
                "text":        results["documents"][0][i],
                "stars":       results["metadatas"][0][i]["stars"],
                "business_id": results["metadatas"][0][i]["business_id"],
                "date":        results["metadatas"][0][i]["date"]
            }
            for i in range(len(results["documents"][0]))
        ]
    else:
        results = _reviews_col.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
            limit=k
        )
        return [
            {
                "text":        results["documents"][i],
                "stars":       results["metadatas"][i]["stars"],
                "business_id": results["metadatas"][i]["business_id"],
                "date":        results["metadatas"][i]["date"]
            }
            for i in range(len(results["documents"]))
        ]

# ── 4. Semantic business search ───────────────────────────────────────────────
def search_businesses(query: str, k: int = 20, min_stars: float = 3.0, city: str = "Lagos") -> list[dict]:
    """
    Find businesses semantically similar to a query string.
    e.g. "spicy Nigerian food casual Lagos" → returns relevant businesses.
    """
    _init()
    embedding = _embedder.encode(query).tolist()
    results   = _business_col.query(
        query_embeddings=[embedding],
        n_results=k,
        where={
            "$and": [
                {"stars": {"$gte": min_stars}},
                {"city": city}
            ]
        },
        include=["metadatas", "distances"]
    )
    return [
        {
            "business_id":     meta["business_id"],
            "name":            meta["name"],
            "categories":      meta["categories"],
            "city":            meta["city"],
            "state":           meta["state"],
            "stars":           meta["stars"],
            "review_count":    meta["review_count"],
            "relevance_score": round(1 - results["distances"][0][i], 4)
        }
        for i, meta in enumerate(results["metadatas"][0])
    ]


# ── 5. Fetch one specific business by ID ──────────────────────────────────────
def get_business(business_id: str) -> dict | None:
    """Fetch a single business's metadata by ID."""
    _init()
    try:
        result = _business_col.get(
            ids=[business_id],
            include=["metadatas"]
        )
        return result["metadatas"][0] if result["ids"] else None
    except Exception:
        return None