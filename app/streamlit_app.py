"""
streamlit_app.py
----------------
Semantic search demo: compare Keyword (BM25), Semantic, and Hybrid (RRF)
search side by side over a product catalog.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys

# Silence noisy (harmless) TensorFlow / CUDA registration logs before imports.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "3")

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from search_engine import SearchEngine  # noqa: E402

st.set_page_config(
    page_title="FindWise — Semantic Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#1F3864"
MINT = "#9FE1CB"
MINT_DEEP = "#0F6E56"

st.markdown(
    f"""
    <style>
    .stApp {{ background:#F7F8FA; }}
    .brand-header {{
        background:{NAVY}; border-radius:14px; padding:20px 26px; margin-bottom:18px;
        display:flex; align-items:center; justify-content:space-between;
    }}
    .brand-left {{ display:flex; align-items:center; gap:12px; }}
    .brand-badge {{
        width:40px; height:40px; border-radius:10px; background:{MINT};
        display:flex; align-items:center; justify-content:center; font-size:22px;
    }}
    .brand-name {{ color:#fff; font-size:20px; font-weight:600; margin:0; }}
    .brand-sub {{ color:{MINT}; font-size:13px; margin:0; }}
    .col-head {{
        font-size:15px; font-weight:600; padding:10px 14px; border-radius:10px 10px 0 0;
        color:#fff; margin:0;
    }}
    .col-note {{ font-size:12px; color:#667085; margin:2px 0 10px; }}
    .card {{
        background:#fff; border:1px solid #E4E7EC; border-radius:10px;
        padding:12px 14px; margin-bottom:10px;
    }}
    .card-top {{
        background:#fff; border:1px solid {MINT}; border-left:4px solid {MINT_DEEP};
        border-radius:0; padding:12px 14px; margin-bottom:10px;
    }}
    .p-title {{ font-size:14px; font-weight:600; color:#1D2939; margin:0 0 3px; }}
    .p-desc {{ font-size:12px; color:#667085; margin:0 0 6px; line-height:1.4; }}
    .p-meta {{ font-size:12px; color:{MINT_DEEP}; font-weight:500; }}
    .rank-num {{
        display:inline-block; width:20px; height:20px; border-radius:5px;
        background:#F1EFE8; color:#5F5E5A; font-size:11px; font-weight:600;
        text-align:center; line-height:20px; margin-right:6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="brand-header">
      <div class="brand-left">
        <div class="brand-badge">🔎</div>
        <div>
          <p class="brand-name">FindWise — Search That Understands Meaning</p>
          <p class="brand-sub">Farwa Khizar · keyword vs semantic vs hybrid, side by side</p>
        </div>
      </div>
      <span style="background:#E1F5EE; color:{MINT_DEEP}; font-size:13px; font-weight:600; padding:6px 14px; border-radius:999px;">● Semantic + Hybrid Re-ranking</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading search engine…")
def get_engine() -> SearchEngine:
    e = SearchEngine()
    data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "products.csv")
    e.load_csv(data)
    return e


engine = get_engine()

# ------------------------------ Side panel --------------------------------- #

with st.sidebar:
    st.markdown("### ⚙️ Search settings")

    top_k = st.slider(
        "Results per method", min_value=3, max_value=10, value=4,
        help="How many results each search method returns.",
    )

    st.markdown("#### Hybrid fusion weights")
    st.caption(
        "Hybrid search fuses the keyword and semantic rankings (Reciprocal "
        "Rank Fusion). These weights control how much each side contributes."
    )
    w_keyword = st.slider("Keyword weight", 0.0, 3.0, 1.0, 0.5)
    w_semantic = st.slider("Semantic weight", 0.0, 3.0, 2.0, 0.5)

    st.divider()
    st.caption(
        f"📦 Catalog: **{engine.size:,} products**  \n"
        "🧠 Embeddings: sentence-transformers (local)  \n"
        "🔎 Keyword: BM25 · Fusion: RRF"
    )
    st.caption("Tip: push semantic weight up and watch hybrid follow meaning; push keyword up and it favors exact terms.")

# ------------------------------ Search bar --------------------------------- #

st.markdown("**Search the catalog in natural language** — see how keyword, semantic, and hybrid rank results differently.")

query = st.text_input(
    "Search the product catalog",
    value="",
    placeholder="e.g. cheap camera for making videos",
    label_visibility="collapsed",
)

COLORS = {"Keyword (BM25)": "#888780", "Semantic": "#185FA5", "Hybrid (RRF)": "#0F6E56"}
NOTES = {
    "Keyword (BM25)": "Classic term matching — literal, misses synonyms.",
    "Semantic": "Matches by meaning using embeddings.",
    "Hybrid (RRF)": "Fuses keyword + semantic, re-ranked.",
}

if query:
    # run each method, honoring the sidebar settings
    results = {
        "Keyword (BM25)": engine.keyword_search(query, k=top_k),
        "Semantic": engine.semantic_search(query, k=top_k),
        "Hybrid (RRF)": engine.hybrid_search(
            query, k=top_k, w_keyword=w_keyword, w_semantic=w_semantic
        ),
    }
    cols = st.columns(3)
    for col, (method, hits) in zip(cols, results.items()):
        with col:
            st.markdown(
                f'<p class="col-head" style="background:{COLORS[method]}">{method}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p class="col-note">{NOTES[method]}</p>', unsafe_allow_html=True)
            for h in hits:
                p = h.product
                card_cls = "card-top" if h.rank == 1 else "card"
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<p class="p-title"><span class="rank-num">{h.rank}</span>{p.title}</p>'
                    f'<p class="p-desc">{p.description[:90]}…</p>'
                    f'<span class="p-meta">${p.price:.0f} · {p.category} · {h.score:.0%} match</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
else:
    st.info("👆 Type a natural-language query to see all three search methods side by side. Adjust the settings in the sidebar to see how the rankings change.")
