import streamlit as st
import pandas as pd
import time
import os
import sys
from pathlib import Path

# Ensure the 'src' directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.search.vector_search import VectorSearchEngine
from src.search.hybrid_search import HybridSearchEngine
from src.monitoring.drift_detector import DriftDetector

# Page config
st.set_page_config(
    page_title="Semantic ArXiv Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize engines with caching to avoid reloading on every interaction
@st.cache_resource
def init_engines():
    v_engine = VectorSearchEngine()
    v_engine.load_index()
    
    h_engine = HybridSearchEngine(v_engine)
    h_engine.load_corpus()
    
    # Initialize DriftDetector with corpus embeddings
    # We use v_engine.index to get embeddings if needed, but here we'll just re-encode a sample or use pre-saved ones
    # Simplified: Encode all abstracts once for the baseline (in a real app, load from disk)
    texts = [f"{p.title}. {p.abstract}" for p in v_engine.papers]
    corpus_embeddings = v_engine.model.encode(texts[:500]) # Sample 500 for baseline speed
    drift_detector = DriftDetector(v_engine.model, corpus_embeddings)
    
    return v_engine, h_engine, drift_detector

v_engine, h_engine, drift_detector = init_engines()

# Sidebar
st.sidebar.title("⚙️ Search Settings")
search_mode = st.sidebar.radio("Search Mode", ["Semantic Only", "Hybrid (BM25 + Semantic)", "Semantic + Reranking"])
top_k = st.sidebar.slider("Number of results", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.title("📊 MLOps: Drift Monitoring")
if "queries" not in st.session_state:
    st.session_state.queries = []

drift_status = drift_detector.check_drift()
st.sidebar.metric("Drift Score", f"{drift_status['drift_score']:.4f}")
if drift_status["is_drifting"]:
    st.sidebar.error("⚠️ Significant Query Drift!")
else:
    st.sidebar.success("✅ Query Distribution Stable")

st.sidebar.markdown("---")
show_details = st.sidebar.checkbox("Show Detailed Scores", value=True)

# Main UI
st.title("📚 Semantic Research Paper Search")
st.markdown("Discover papers based on meaning, not just keywords.")

query = st.text_input("Enter your research topic or question:", placeholder="e.g., 'Transformer models for time series forecasting'")

if query:
    # Track query for drift
    drift_detector.add_query(query)
    
    with st.spinner("Searching..."):
        start_time = time.time()
        
        if search_mode == "Semantic Only":
            results = v_engine.search(query, top_k=top_k)
        elif search_mode == "Hybrid (BM25 + Semantic)":
            results = h_engine.hybrid_search(query, top_k=top_k)
        else: # Semantic + Reranking
            initial_results = h_engine.hybrid_search(query, top_k=20)
            results = h_engine.rerank(query, initial_results)[:top_k]
            
        latency = (time.time() - start_time) * 1000

    st.success(f"Found {len(results)} papers in {latency:.2f}ms")

    # Display results
    for i, res in enumerate(results):
        with st.container():
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.subheader(f"{i+1}. {res.title}")
            with col2:
                # Primary Score
                st.markdown(f"**Final Score: `{res.score:.4f}`**")
                
                # Detailed Scores if enabled
                if show_details:
                    score_details = []
                # Detailed Scores if enabled
                if show_details:
                    score_details = []
                    # Use getattr as a safety measure against caching/serialization issues
                    s_score = getattr(res, 'semantic_score', None)
                    k_score = getattr(res, 'keyword_score', None)
                    r_score = getattr(res, 'rrf_score', None)
                    re_score = getattr(res, 'rerank_score', None)

                    if s_score is not None:
                        score_details.append(f"Similarity: `{s_score:.4f}`")
                    if k_score is not None:
                        score_details.append(f"BM25: `{k_score:.4f}`")
                    if r_score is not None:
                        score_details.append(f"RRF: `{r_score:.4f}`")
                    if re_score is not None:
                        score_details.append(f"Cross-Encoder: `{re_score:.4f}`")
                    
                    if score_details:
                        st.caption(" | ".join(score_details))
            
            st.markdown(f"*Authors: {', '.join(res.authors)} | {res.published_date}*")
            st.write(res.abstract)
            
            # Categories and Links
            tags = " ".join([f"`{c}`" for c in res.categories[:3]])
            st.markdown(f"{tags} | [Read Paper]({res.url})")
            st.markdown("---")

else:
    st.info("💡 Try searching for 'Deep learning in healthcare' or 'Climate change modeling'")

# Analytics Dashboard (Optional expander)
with st.expander("🔬 View Search Pipeline Metadata"):
    st.write(f"**Total Papers Indexed:** {len(v_engine.papers)}")
    st.write(f"**Vector Engine:** FAISS (L2 + Normalization)")
    st.write(f"**Embeddings:** finetuned_model_fixed (Base: all-MiniLM-L6-v2)")
    st.write(f"**Data Contracts:** Enforced via Pydantic")
