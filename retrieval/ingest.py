import os
import sys
import json
import chromadb
import numpy as np

from sentence_transformers import SentenceTransformer
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ── Paths (works regardless of where you run the script from) ─────────────────
ROOT         = Path(__file__).parent.parent
PROCESSED    = ROOT / "data" / "processed"
CHROMA_DIR   = str(ROOT / "chroma_db")

# ── Config ────────────────────────────────────────────────────────────────────
REVIEWS_PER_USER = 30   # only embed N most recent reviews per user
BATCH_SIZE       = 512

# ── Init embedding model + ChromaDB ──────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client       = chromadb.PersistentClient(path=CHROMA_DIR)
personas_col = client.get_or_create_collection("user_personas")
reviews_col  = client.get_or_create_collection("user_reviews")
business_col = client.get_or_create_collection("businesses")

# ── Load processed data ───────────────────────────────────────────────────────
print("Loading reviews...")
user_reviews = defaultdict(list)
with open(PROCESSED / "review_slice.json", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        user_reviews[r["user_id"]].append(r)
print(f"  {sum(len(v) for v in user_reviews.values())} reviews | {len(user_reviews)} users")

print("Loading businesses...")
businesses = {}
with open(PROCESSED / "business_slice.json", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        b = json.loads(line)
        if b.get("is_open", True) and int(b.get("review_count") or 0) >= 5:
            businesses[b["business_id"]] = b
print(f"  {len(businesses)} Yelp businesses loaded")

naija_path = PROCESSED / "naija_slice.json"
if naija_path.exists():
    with open(naija_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            b = json.loads(line)
            if b.get("is_open", True):
                businesses[b["business_id"]] = b
    print(f"  Nigerian businesses injected ✅")

print(f"  {len(businesses)} total businesses after merge")

# ── Helper: build a persona from a user's reviews ────────────────────────────
def build_persona(user_id: str, reviews: list) -> dict:
    ratings = [r["stars"] for r in reviews]
    lengths = [len(r["text"].split()) for r in reviews]

    avg_rating       = round(sum(ratings) / len(ratings), 2)
    rating_variance  = round(float(np.std(ratings)), 2)
    avg_length       = round(sum(lengths) / len(lengths))

    # Tone tags derived from behavior patterns
    tone_tags = []
    if avg_rating >= 4.0:    tone_tags.append("generous_rater")
    elif avg_rating <= 2.5:  tone_tags.append("harsh_rater")
    else:                    tone_tags.append("balanced_rater")

    if avg_length > 100:     tone_tags.append("detailed_writer")
    elif avg_length < 40:    tone_tags.append("brief_writer")

    if rating_variance > 1.2: tone_tags.append("inconsistent_rater")
    else:                      tone_tags.append("consistent_rater")

    # Top business categories this user frequents
    all_cats = []
    for r in reviews:
        biz = businesses.get(r["business_id"])
        if biz and biz.get("categories"):
            all_cats.extend(biz["categories"].split(", "))
    cat_counts = defaultdict(int)
    for c in all_cats:
        cat_counts[c] += 1
    top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda x: -x[1])[:5]]

    # 5 most recent reviews as context snippets
    recent = sorted(reviews, key=lambda r: r["date"], reverse=True)[:5]

    return {
        "user_id":          user_id,
        "avg_rating":       avg_rating,
        "rating_variance":  rating_variance,
        "avg_review_length": avg_length,
        "total_reviews":    len(reviews),
        "tone_tags":        tone_tags,
        "top_categories":   top_cats,
        "recent_reviews": [
            {
                "business_id": r["business_id"],
                "stars":       r["stars"],
                "text":        r["text"][:200],
                "date":        r["date"]
            }
            for r in recent
        ]
    }

# ── 1. Build + store personas ─────────────────────────────────────────────────
print("\nBuilding and storing personas...")
p_ids, p_docs, p_metas = [], [], []

for user_id, reviews in tqdm(user_reviews.items()):
    p = build_persona(user_id, reviews)

    # Human-readable summary — this is what gets embedded
    summary = (
        f"User who rates on average {p['avg_rating']} stars. "
        f"Writes {p['avg_review_length']}-word reviews on average. "
        f"Tone: {', '.join(p['tone_tags'])}. "
        f"Frequently visits: {', '.join(p['top_categories'][:3])}."
    )

    p_ids.append(user_id)
    p_docs.append(summary)
    p_metas.append({
        "user_id":            user_id,
        "avg_rating":         p["avg_rating"],
        "rating_variance":    p["rating_variance"],
        "avg_review_length":  p["avg_review_length"],
        "total_reviews":      p["total_reviews"],
        "tone_tags":          json.dumps(p["tone_tags"]),
        "top_categories":     json.dumps(p["top_categories"]),
        "recent_reviews":     json.dumps(p["recent_reviews"])
    })

for i in range(0, len(p_ids), BATCH_SIZE):
    embeddings = embedder.encode(p_docs[i:i+BATCH_SIZE], show_progress_bar=False).tolist()
    personas_col.upsert(
        ids=p_ids[i:i+BATCH_SIZE],
        documents=p_docs[i:i+BATCH_SIZE],
        metadatas=p_metas[i:i+BATCH_SIZE],
        embeddings=embeddings
    )
print(f"✅ {len(p_ids)} personas stored")


# ── 2. Embed reviews for retrieval ────────────────────────────────────────────
print("\nEmbedding reviews for retrieval...")
r_ids, r_docs, r_metas = [], [], []

for user_id, reviews in user_reviews.items():
    recent = sorted(reviews, key=lambda r: r["date"], reverse=True)[:REVIEWS_PER_USER]
    for r in recent:
        r_ids.append(r["review_id"])
        r_docs.append(r["text"][:512])
        r_metas.append({
            "user_id":     r["user_id"],
            "business_id": r["business_id"],
            "stars":       r["stars"],
            "date":        r["date"]
        })

print(f"  Embedding {len(r_ids)} reviews...")
for i in tqdm(range(0, len(r_ids), BATCH_SIZE)):
    embeddings = embedder.encode(r_docs[i:i+BATCH_SIZE], show_progress_bar=False).tolist()
    reviews_col.upsert(
        ids=r_ids[i:i+BATCH_SIZE],
        documents=r_docs[i:i+BATCH_SIZE],
        metadatas=r_metas[i:i+BATCH_SIZE],
        embeddings=embeddings
    )
print(f"✅ {len(r_ids)} reviews stored")

# ── 3. Embed businesses ───────────────────────────────────────────────────────
print("\nEmbedding businesses...")
b_ids, b_docs, b_metas = [], [], []

for biz_id, biz in businesses.items():
    b_ids.append(biz_id)
    b_docs.append(f"{biz.get('name','')}. {biz.get('categories','')}. {biz.get('city','')}.")
    b_metas.append({
        "business_id":  biz_id,
        "name":         biz.get("name", ""),
        "categories":   biz.get("categories", ""),
        "city":         biz.get("city", ""),
        "state":        biz.get("state", ""),
        "stars":        float(biz.get("stars", 0)),
        "review_count": int(biz.get("review_count", 0)),
    })

for i in tqdm(range(0, len(b_ids), BATCH_SIZE)):
    embeddings = embedder.encode(b_docs[i:i+BATCH_SIZE], show_progress_bar=False).tolist()
    business_col.upsert(
        ids=b_ids[i:i+BATCH_SIZE],
        documents=b_docs[i:i+BATCH_SIZE],
        metadatas=b_metas[i:i+BATCH_SIZE],
        embeddings=embeddings
    )
print(f"✅ {len(b_ids)} businesses stored")

print("\n🚀 Ingestion complete. ChromaDB is ready.")