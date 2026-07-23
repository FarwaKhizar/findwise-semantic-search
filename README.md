# 🔎 FindWise — Semantic Search (Keyword vs Semantic vs Hybrid)

**Search that understands meaning, not just keywords — shown side by side against traditional search.**

FindWise runs three search methods on the same query at once so you can *see* the difference:

- **Keyword (BM25)** — classic term matching. Fast, but literal — misses synonyms and intent.
- **Semantic** — matches by *meaning* using embeddings, so "keep my drinks warm" finds a thermostat.
- **Hybrid (RRF)** — fuses keyword + semantic rankings and re-ranks, combining exact-term precision with semantic understanding.

> Built by **Farwa Khizar** — AI / NLP Engineer specializing in semantic search, retrieval, and ranking. Built on the same techniques used to run search over 100M+ vectors in production.

---

## ✨ See the difference

Try a query like **"keep my drinks warm at home"**:

| Method | Top result | Why |
|---|---|---|
| Keyword | ❌ LED Desk Lamp | matched only the word "warm" |
| Semantic | ✅ Smart Thermostat | understood you mean *temperature control* |
| Hybrid | ✅ Smart Thermostat | semantic intent + keyword precision |

Or **"cheap camera for making videos"** — keyword grabs the wrong camera on the word "camera"; semantic and hybrid surface the vlogging camera that actually fits the intent.

---

## 🧱 How it works

```
             ┌─ Keyword (BM25) ──────► ranked list ─┐
query ──────►│                                      ├─► RRF fusion ─► re-ranked results
             └─ Semantic (embeddings) ► ranked list ─┘
```

| Component | Tool | Notes |
|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Local, free, no API key |
| Keyword ranking | BM25 (Okapi) | Compact in-repo implementation |
| Fusion | Reciprocal Rank Fusion (RRF) | Robust to differing score scales; semantic-weighted |
| UI | Streamlit | Three-column side-by-side comparison |

**Everything runs locally and offline — no API keys, no cost, no rate limits.** In production the same pattern maps to Elasticsearch/OpenSearch BM25 + a vector index (Pinecone, Qdrant, Weaviate).

---

## 📊 Using your own dataset

The app reads `data/products.csv`. To swap in a larger catalog (e.g. an Amazon/e-commerce dataset from Kaggle), format your CSV with these exact columns:

| Column | Type | Notes |
|---|---|---|
| `id` | integer | Unique per product |
| `title` | text | Product name |
| `description` | text | **Important** — semantic search quality depends on descriptive text, not just titles |
| `category` | text | Product category |
| `price` | number | Numeric price |

Then replace `data/products.csv` and restart the app. For large files, sampling down to ~10,000–20,000 rows keeps the one-time embedding step fast (a couple of minutes) while still looking substantial in the UI. Quote any description fields that contain commas.

---

## ⚙️ Interactive settings

The sidebar lets you tune search live:
- **Results per method** — how many results each column shows.
- **Fusion weights** — how much the keyword vs semantic ranking contributes to the hybrid result. Push semantic up and hybrid follows intent; push keyword up and it favors exact terms.

---

## 🚀 Quick start

```bash
git clone <your-repo-url>
cd semantic-search-demo
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then click an example query or type your own, and watch the three methods rank results differently.

---

## 📁 Project structure

```
semantic-search-demo/
├── app/
│   ├── search_engine.py    # BM25 + semantic + hybrid (RRF) search
│   └── streamlit_app.py    # Three-column comparison UI
├── data/
│   └── products.csv        # Curated 30-item product catalog
├── requirements.txt
└── README.md
```

---

## 🔧 Production notes

- **Scale** — swap the in-memory index for a vector DB (Pinecone/Qdrant/Weaviate) and BM25 for Elasticsearch/OpenSearch; the interfaces stay the same. This pattern has been used on retrieval over 100M+ vectors.
- **Fusion weighting** — `hybrid_search()` exposes `w_keyword` / `w_semantic` weights; tune per domain. A cross-encoder re-ranker can be added on top of the fused list for higher precision.
- **Evaluation** — add a labeled query→relevant-item set to measure precision@k / nDCG as you tune.

---

## 📬 Work with me

I build production semantic search, hybrid retrieval, and ranking systems. If your search returns irrelevant results or misses what users mean, let's talk.
