import sys
import os
import json
import time
from pathlib import Path
from collections import Counter
import re

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import joblib

from src.database import (
    get_prediction_history,
    log_prediction,
    submit_feedback,
    add_custom_question,
    get_combined_dataset,
    get_custom_questions,
)
from src.model_training import ModelTrainer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DifficultyAI · JEE/NEET Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-card: rgba(17, 17, 27, 0.85);
        --bg-card-hover: rgba(25, 25, 40, 0.95);
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-hover: rgba(99, 102, 241, 0.4);
        --text-primary: #e4e4e7;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent-indigo: #6366f1;
        --accent-violet: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
        --gradient-warm: linear-gradient(135deg, #f43f5e 0%, #f59e0b 100%);
        --gradient-cool: linear-gradient(135deg, #06b6d4 0%, #6366f1 100%);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.4);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
        --radius-lg: 16px;
        --radius-md: 12px;
        --radius-sm: 8px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Base Overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(99, 102, 241, 0.12), transparent),
            radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139, 92, 246, 0.08), transparent);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 18, 0.95) !important;
        border-right: 1px solid var(--border-subtle) !important;
        backdrop-filter: blur(20px);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: var(--text-secondary) !important;
        font-weight: 500;
    }

    /* ── Headers ── */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: var(--text-muted);
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.01em;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-card);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }
    .glass-card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--shadow-card), var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-card);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0;
        width: 100%;
        height: 3px;
    }
    .metric-card.indigo::after { background: var(--gradient-primary); }
    .metric-card.cyan::after { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
    .metric-card.emerald::after { background: linear-gradient(90deg, #10b981, #34d399); }
    .metric-card.amber::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .metric-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-3px);
        box-shadow: var(--shadow-card), var(--shadow-glow);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .metric-value.indigo { color: var(--accent-indigo); }
    .metric-value.cyan { color: var(--accent-cyan); }
    .metric-value.emerald { color: var(--accent-emerald); }
    .metric-value.amber { color: var(--accent-amber); }
    .metric-delta {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* ── Difficulty Badges ── */
    .badge-easy {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 0.5rem 1.2rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
    }
    .badge-medium {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fbbf24;
        padding: 0.5rem 1.2rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
    }
    .badge-hard {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(244, 63, 94, 0.1);
        border: 1px solid rgba(244, 63, 94, 0.3);
        color: #fb7185;
        padding: 0.5rem 1.2rem;
        border-radius: 100px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
    }

    /* ── Prediction Result Box ── */
    .result-box {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    .confidence-big {
        font-size: 3.5rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .confidence-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        font-weight: 600;
    }

    /* ── Tag Pills ── */
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 0.2rem 0.7rem;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    /* ── Model Table ── */
    .model-row {
        display: flex;
        align-items: center;
        padding: 0.9rem 1.2rem;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        margin-bottom: 0.5rem;
        transition: var(--transition);
        gap: 1rem;
    }
    .model-row:hover {
        border-color: var(--border-hover);
        background: var(--bg-card-hover);
    }
    .model-row.best {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(99, 102, 241, 0.05);
    }
    .model-name {
        font-weight: 600;
        color: var(--text-primary);
        width: 180px;
        flex-shrink: 0;
    }
    .model-stat {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 500;
        width: 90px;
        text-align: center;
    }
    .model-stat-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        display: block;
        margin-bottom: 2px;
    }
    .best-badge {
        background: var(--gradient-primary);
        color: white;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-left: auto;
    }

    /* ── Prediction History Cards ── */
    .history-item {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: var(--transition);
    }
    .history-item:hover {
        border-color: var(--border-hover);
    }
    .history-q {
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.4;
        margin-bottom: 0.5rem;
    }
    .history-meta {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        flex-wrap: wrap;
    }

    /* ── Pipeline Section ── */
    .pipeline-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .pipeline-icon {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        border: 1px solid var(--border-subtle);
    }
    .pipeline-icon.step1 { background: rgba(99,102,241,0.15); }
    .pipeline-icon.step2 { background: rgba(6,182,212,0.15); }
    .pipeline-icon.step3 { background: rgba(16,185,129,0.15); }
    .pipeline-icon.step4 { background: rgba(245,158,11,0.15); }
    .pipeline-icon.step5 { background: rgba(244,63,94,0.15); }
    .pipeline-text h4 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .pipeline-text p {
        margin: 0.2rem 0 0;
        font-size: 0.8rem;
        color: var(--text-muted);
        line-height: 1.4;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: var(--text-muted);
        font-size: 0.75rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 3rem;
    }

    /* ── Streamlit Overrides ── */
    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.5rem !important;
        transition: var(--transition) !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    }
    .stTextArea textarea, .stSelectbox > div > div {
        background: rgba(17, 17, 27, 0.8) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus, .stSelectbox > div > div:focus-within {
        border-color: var(--accent-indigo) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    div[data-testid="stForm"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.5rem !important;
    }
    .stProgress > div > div > div {
        background: var(--gradient-primary) !important;
        border-radius: 100px !important;
    }
    .stSpinner > div { color: var(--accent-indigo) !important; }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: var(--border-subtle);
        margin: 2rem 0;
    }
    
    /* Hide default streamlit footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_metrics():
    metrics_path = project_root / "models" / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def dark_fig(figsize=(7, 4)):
    """Create a matplotlib figure with a consistent dark theme."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0f')
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#71717a', labelsize=9)
    ax.xaxis.label.set_color('#a1a1aa')
    ax.yaxis.label.set_color('#a1a1aa')
    ax.title.set_color('#e4e4e7')
    ax.title.set_fontsize(11)
    ax.title.set_fontweight(600)
    for spine in ax.spines.values():
        spine.set_color('#27272a')
    ax.grid(True, alpha=0.08, color='#ffffff')
    return fig, ax


PALETTE = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6',
           '#ec4899', '#14b8a6', '#a78bfa', '#fbbf24']
DIFFICULTY_COLORS = {'Easy': '#10b981', 'Medium': '#f59e0b', 'Hard': '#f43f5e'}


# Subject → Topics
SUBJECT_TOPICS = {
    "Physics": ["Mechanics", "Electrodynamics", "Thermodynamics", "Modern Physics", "Optics"],
    "Chemistry": ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry"],
    "Mathematics": ["Calculus", "Algebra", "Coordinate Geometry", "Trigonometry", "Vectors & 3D"],
    "Biology": ["Genetics", "Ecology", "Human Physiology", "Plant Physiology", "Cell Biology"],
}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.2rem;">🧠</div>
        <div style="font-size:1.1rem; font-weight:800; background:linear-gradient(135deg,#6366f1,#8b5cf6);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top:0.3rem;">
            DifficultyAI
        </div>
        <div style="font-size:0.7rem; color:#71717a; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.15rem;">
            JEE / NEET Predictor v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview", "📊  Dataset Analytics", "⚙️  Model Lab", "🔮  Predict Difficulty"],
        index=3,
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    metrics = load_metrics()

    if metrics:
        best_name = metrics.get("best_model", "N/A")
        best_acc = metrics.get(best_name, {}).get("accuracy", 0)
        best_f1 = metrics.get(best_name, {}).get("f1_score", 0)
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2);
             border-radius:12px; padding:1rem; text-align:center;">
            <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:#71717a; font-weight:600;">
                Active Model
            </div>
            <div style="font-size:1.1rem; font-weight:700; color:#a5b4fc; margin:0.3rem 0 0.15rem;">
                {best_name}
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:#6366f1;">
                {best_acc*100:.1f}% acc · {best_f1*100:.1f}% F1
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No trained model")

    st.markdown("""
    <div class="footer" style="margin-top:2rem; border:none; padding-top:0;">
        Built with Scikit-learn · XGBoost · NLTK<br>
        Streamlit Dashboard · Python 3.10+
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero-title">JEE/NEET Question<br>Difficulty Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML-powered classification engine that predicts exam question difficulty using NLP feature engineering and ensemble learning</div>', unsafe_allow_html=True)

    # ── Metric Row ──
    if metrics:
        ds = get_combined_dataset()
        best_name = metrics.get("best_model", "")
        best_metrics = metrics.get(best_name, {})
        dm = metrics.get("dataset_metrics", {})

        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(f"""<div class="metric-card indigo">
                <div class="metric-label">Dataset Size</div>
                <div class="metric-value indigo">{len(ds):,}</div>
                <div class="metric-delta">questions across 4 subjects</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card cyan">
                <div class="metric-label">Test Accuracy</div>
                <div class="metric-value cyan">{best_metrics.get('accuracy',0)*100:.1f}%</div>
                <div class="metric-delta">on held-out test split</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card emerald">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value emerald">{best_metrics.get('f1_score',0)*100:.1f}%</div>
                <div class="metric-delta">weighted macro average</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card amber">
                <div class="metric-label">Feature Dimensions</div>
                <div class="metric-value amber">{dm.get('num_features',0):,}</div>
                <div class="metric-delta">TF-IDF + categorical + numerical</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── ML Pipeline ──
    col_pipe, col_roadmap = st.columns([1, 1], gap="large")

    with col_pipe:
        st.markdown('<div class="section-title">⚡ ML Pipeline Architecture</div>', unsafe_allow_html=True)
        steps = [
            ("step1", "📥", "Data Ingestion", "Load 510+ JEE/NEET questions from CSV + SQLite custom entries"),
            ("step2", "🧹", "Text Preprocessing", "NLTK tokenization · stopword removal · WordNet lemmatization"),
            ("step3", "🔬", "Feature Engineering", "TF-IDF (1000-dim) + math symbol counts + keyword density + one-hot encoding"),
            ("step4", "🏋️", "Multi-Model Training", "Logistic Regression · Random Forest · SVM · XGBoost comparison"),
            ("step5", "🏆", "Auto-Selection & Deploy", "Best model selected by weighted F1-score, serialized with joblib"),
        ]
        for step_cls, icon, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-icon {step_cls}">{icon}</div>
                <div class="pipeline-text">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_roadmap:
        st.markdown('<div class="section-title">🚀 Future AI Roadmap</div>', unsafe_allow_html=True)
        roadmap = [
            ("🤖", "LLM Integration", "Gemini / GPT-powered step-by-step solution generation for Hard questions"),
            ("📚", "RAG System", "ChromaDB vector store over NCERT textbooks for context-aware concept retrieval"),
            ("🎯", "Study Recommender", "Personalized study plans generated from prediction history & weak-topic analysis"),
            ("⚡", "API Microservice", "FastAPI + Docker deployment for sub-10ms inference at scale"),
        ]
        for icon, title, desc in roadmap:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-icon step1">{icon}</div>
                <div class="pipeline-text">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Tech Stack Tags ──
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    techs = ["Python 3.10+", "Scikit-learn", "XGBoost", "NLTK", "Pandas", "NumPy",
             "Streamlit", "Matplotlib", "Seaborn", "SQLite", "Joblib", "TF-IDF"]
    tag_html = " ".join([f'<span class="tag">{t}</span>' for t in techs])
    st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:0.4rem;">{tag_html}</div>', unsafe_allow_html=True)

    st.markdown("""<div class="footer">
        DifficultyAI © 2025 · Built as a production-grade ML portfolio project
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2 — DATASET ANALYTICS
# ═══════════════════════════════════════════════
elif page == "📊  Dataset Analytics":
    st.markdown('<div class="hero-title">Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploratory data analysis across subjects, topics, and difficulty levels</div>', unsafe_allow_html=True)

    df = get_combined_dataset()
    if df.empty:
        st.error("No data available. Generate the dataset first.")
    else:
        # Quick stats row
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            st.markdown(f"""<div class="metric-card indigo">
                <div class="metric-label">Total Questions</div>
                <div class="metric-value indigo">{len(df)}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card cyan">
                <div class="metric-label">Subjects</div>
                <div class="metric-value cyan">{df['Subject'].nunique()}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card emerald">
                <div class="metric-label">Topics</div>
                <div class="metric-value emerald">{df['Topic'].nunique()}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            avg_len = int(df['Question_Text'].str.len().mean())
            st.markdown(f"""<div class="metric-card amber">
                <div class="metric-label">Avg. Question Length</div>
                <div class="metric-value amber">{avg_len}</div>
                <div class="metric-delta">characters</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Charts row 1
        ch1, ch2 = st.columns(2, gap="medium")

        with ch1:
            st.markdown('<div class="section-title">📚 Subject × Difficulty Distribution</div>', unsafe_allow_html=True)
            fig, ax = dark_fig((7, 4))
            order = ['Easy', 'Medium', 'Hard']
            palette_diff = [DIFFICULTY_COLORS[d] for d in order]
            sns.countplot(data=df, x='Subject', hue='Difficulty', hue_order=order,
                          palette=palette_diff, ax=ax, edgecolor='none')
            ax.legend(frameon=False, labelcolor='#a1a1aa', fontsize=8)
            ax.set_xlabel('')
            ax.set_title('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with ch2:
            st.markdown('<div class="section-title">🎯 Difficulty Balance</div>', unsafe_allow_html=True)
            fig, ax = dark_fig((7, 4))
            counts = df['Difficulty'].value_counts()
            colors_pie = [DIFFICULTY_COLORS.get(d, '#6366f1') for d in counts.index]
            wedges, texts, autotexts = ax.pie(
                counts, labels=counts.index, autopct='%1.1f%%',
                colors=colors_pie, startangle=90,
                textprops={'color': '#e4e4e7', 'fontsize': 10, 'fontweight': 600},
                wedgeprops={'edgecolor': '#0a0a0f', 'linewidth': 2},
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_color('#e4e4e7')
            ax.set_title('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Charts row 2
        ch3, ch4 = st.columns(2, gap="medium")

        with ch3:
            st.markdown('<div class="section-title">📝 Question Length by Difficulty</div>', unsafe_allow_html=True)
            fig, ax = dark_fig((7, 4))
            df['_charlen'] = df['Question_Text'].str.len()
            sns.boxplot(data=df, x='Difficulty', y='_charlen', order=order,
                        palette=[DIFFICULTY_COLORS[d] for d in order], ax=ax,
                        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': '#71717a'})
            ax.set_ylabel('Character Count')
            ax.set_xlabel('')
            ax.set_title('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with ch4:
            st.markdown('<div class="section-title">🔤 Top 15 Keywords (Corpus-wide)</div>', unsafe_allow_html=True)
            all_text = " ".join(df['Question_Text'].dropna().tolist()).lower()
            words = re.findall(r'\b[a-z]{3,}\b', all_text)
            try:
                from nltk.corpus import stopwords as sw
                stop_words = set(sw.words('english'))
            except Exception:
                stop_words = set()
            stop_words |= {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'what',
                           'find', 'calculate', 'determine', 'value', 'given', 'which',
                           'following', 'show', 'write', 'state', 'define', 'name', 'explain'}
            filtered = [w for w in words if w not in stop_words]
            top15 = Counter(filtered).most_common(15)
            if top15:
                wdf = pd.DataFrame(top15, columns=['Word', 'Freq'])
                fig, ax = dark_fig((7, 4))
                sns.barplot(data=wdf, y='Word', x='Freq', palette=PALETTE[:15], ax=ax, edgecolor='none')
                ax.set_ylabel('')
                ax.set_xlabel('Frequency')
                ax.set_title('')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Data preview
        st.markdown('<div class="section-title">📄 Dataset Preview</div>', unsafe_allow_html=True)
        st.dataframe(
            df.head(8)[["Question_Text", "Subject", "Topic", "Difficulty"]],
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════
# PAGE 3 — MODEL LAB
# ═══════════════════════════════════════════════
elif page == "⚙️  Model Lab":
    st.markdown('<div class="hero-title">Model Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Training metrics, confusion matrices, feature importances, and on-demand retraining</div>', unsafe_allow_html=True)

    if not metrics:
        st.warning("No training metrics found. Train models using the button below.")
    else:
        best_name = metrics.get("best_model", "")

        # ── Model Comparison Rows ──
        st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)

        for mname in ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]:
            m = metrics.get(mname, {})
            if not m:
                continue
            is_best = mname == best_name
            row_cls = "model-row best" if is_best else "model-row"
            badge = '<span class="best-badge">🏆 Best</span>' if is_best else ''

            st.markdown(f"""
            <div class="{row_cls}">
                <div class="model-name">{'⭐ ' if is_best else ''}{mname}</div>
                <div class="model-stat">
                    <span class="model-stat-label">Accuracy</span>
                    <span style="color:{'#6366f1' if is_best else '#a1a1aa'}">{m['accuracy']*100:.1f}%</span>
                </div>
                <div class="model-stat">
                    <span class="model-stat-label">Precision</span>
                    <span style="color:{'#06b6d4' if is_best else '#a1a1aa'}">{m['precision']*100:.1f}%</span>
                </div>
                <div class="model-stat">
                    <span class="model-stat-label">Recall</span>
                    <span style="color:{'#10b981' if is_best else '#a1a1aa'}">{m['recall']*100:.1f}%</span>
                </div>
                <div class="model-stat">
                    <span class="model-stat-label">F1 Score</span>
                    <span style="color:{'#f59e0b' if is_best else '#a1a1aa'}">{m['f1_score']*100:.1f}%</span>
                </div>
                {badge}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Charts
        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown(f'<div class="section-title">🧩 Confusion Matrix · {best_name}</div>', unsafe_allow_html=True)
            cm = np.array(metrics[best_name]["confusion_matrix"])
            fig, ax = dark_fig((6, 5))
            labels = ["Easy", "Medium", "Hard"]
            sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=labels, yticklabels=labels, ax=ax,
                        linewidths=0.5, linecolor='#1a1a24',
                        annot_kws={"fontsize": 14, "fontweight": "bold", "color": "#e4e4e7"},
                        cbar_kws={"shrink": 0.8})
            ax.set_xlabel('Predicted', fontsize=10, color='#a1a1aa')
            ax.set_ylabel('Actual', fontsize=10, color='#a1a1aa')
            ax.set_title('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with ch2:
            st.markdown('<div class="section-title">💡 Feature Importance (Top 12)</div>', unsafe_allow_html=True)
            try:
                fe = joblib.load(project_root / "models" / "feature_engineer.joblib")
                model = joblib.load(project_root / "models" / "best_model.joblib")

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_names = fe.get_feature_names()
                    indices = np.argsort(importances)[::-1][:12]
                    top_imp = importances[indices]
                    top_names = [feat_names[i].replace("tfidf_", "").replace("cat_", "").replace("_", " ") for i in indices]

                    fig, ax = dark_fig((6, 5))
                    bars = ax.barh(range(len(top_names)), top_imp[::-1], color=PALETTE[:12], edgecolor='none')
                    ax.set_yticks(range(len(top_names)))
                    ax.set_yticklabels(top_names[::-1], fontsize=8)
                    ax.set_xlabel('Importance', fontsize=10)
                    ax.set_title('')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                elif hasattr(model, 'coef_'):
                    coefs = np.abs(model.coef_).mean(axis=0)
                    feat_names = fe.get_feature_names()
                    indices = np.argsort(coefs)[::-1][:12]
                    top_c = coefs[indices]
                    top_names = [feat_names[i].replace("tfidf_", "").replace("cat_", "") for i in indices]

                    fig, ax = dark_fig((6, 5))
                    ax.barh(range(len(top_names)), top_c[::-1], color=PALETTE[:12])
                    ax.set_yticks(range(len(top_names)))
                    ax.set_yticklabels(top_names[::-1], fontsize=8)
                    ax.set_xlabel('Avg Coefficient Weight')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.error(f"Could not load importance: {e}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Accuracy bar chart ──
        st.markdown('<div class="section-title">📈 Accuracy Comparison Chart</div>', unsafe_allow_html=True)
        model_names = []
        model_accs = []
        for mname in ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]:
            m = metrics.get(mname, {})
            if m:
                model_names.append(mname)
                model_accs.append(m['accuracy'] * 100)

        fig, ax = dark_fig((10, 3.5))
        colors_bar = ['#6366f1' if n == best_name else '#3f3f46' for n in model_names]
        bars = ax.barh(model_names, model_accs, color=colors_bar, height=0.5, edgecolor='none')
        for bar, acc in zip(bars, model_accs):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1f}%', va='center', ha='left', color='#a1a1aa',
                    fontsize=10, fontweight=600, fontfamily='JetBrains Mono')
        ax.set_xlim(0, 100)
        ax.set_xlabel('')
        ax.set_title('')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Retrain ──
    st.markdown('<div class="section-title">🔄 Retrain Pipeline</div>', unsafe_allow_html=True)
    custom_df = get_custom_questions()
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:1rem;">
        <span style="color:var(--text-muted); font-size:0.85rem;">
            Retraining compiles all baseline questions + <strong style="color:var(--accent-indigo);">{len(custom_df)} custom questions</strong>
            from SQLite, re-engineers features, trains 4 classifiers, and auto-deploys the best model.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Trigger Full Retraining Pipeline", type="primary", use_container_width=True):
        with st.spinner("Training in progress — preprocessing → feature engineering → model training → evaluation ..."):
            try:
                from main import run_training
                run_training()
                st.success("✅ Models retrained successfully! Refresh the page to see updated metrics.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")


# ═══════════════════════════════════════════════
# PAGE 4 — PREDICT DIFFICULTY
# ═══════════════════════════════════════════════
elif page == "🔮  Predict Difficulty":
    st.markdown('<div class="hero-title">Predict Difficulty</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter any JEE or NEET question to classify its difficulty level with confidence scoring</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown('<div class="section-title">📝 Question Input</div>', unsafe_allow_html=True)

        question_text = st.text_area(
            "Enter your question:",
            value="What is the derivative of x² + 5x + 2?",
            height=130,
            placeholder="Type or paste a JEE/NEET exam question...",
            label_visibility="collapsed",
        )

        c_subj, c_topic = st.columns(2)
        with c_subj:
            selected_subject = st.selectbox("Subject", list(SUBJECT_TOPICS.keys()))
        with c_topic:
            selected_topic = st.selectbox("Topic", SUBJECT_TOPICS[selected_subject])

        predict_btn = st.button("🔍  Analyze & Predict", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-title">📊 Classification Result</div>', unsafe_allow_html=True)

        if predict_btn or "last_prediction" in st.session_state:
            if predict_btn:
                if not question_text.strip():
                    st.error("Please enter a valid question.")
                    st.stop()
                with st.spinner("Analyzing linguistic features & mathematical complexity..."):
                    try:
                        trainer = ModelTrainer()
                        result = trainer.predict_single(question_text, selected_subject, selected_topic)
                        pred_id = log_prediction(
                            question_text, selected_subject, selected_topic,
                            result["difficulty"], result["confidence"],
                        )
                        st.session_state["last_prediction"] = {
                            "id": pred_id,
                            "question": question_text,
                            "subject": selected_subject,
                            "topic": selected_topic,
                            "difficulty": result["difficulty"],
                            "confidence": result["confidence"],
                            "probabilities": result.get("probabilities", {}),
                        }
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        st.stop()

            if "last_prediction" in st.session_state:
                pred = st.session_state["last_prediction"]
                diff = pred["difficulty"]
                conf = pred["confidence"]

                badge_cls = f"badge-{diff.lower()}"

                st.markdown(f"""
                <div class="result-box">
                    <div class="confidence-label">Predicted Difficulty</div>
                    <div style="margin: 0.8rem 0;">
                        <span class="{badge_cls}">{diff.upper()}</span>
                    </div>
                    <div class="confidence-label" style="margin-top:1.2rem;">Model Confidence</div>
                    <div class="confidence-big">{conf*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                st.progress(conf)

                # Probability distribution
                probs = pred.get("probabilities", {})
                if probs:
                    fig, ax = dark_fig((6, 2.5))
                    prob_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    names = [p[0] for p in prob_items]
                    vals = [p[1] for p in prob_items]
                    colors = [DIFFICULTY_COLORS.get(n, '#6366f1') for n in names]
                    bars = ax.barh(names, vals, color=colors, height=0.5, edgecolor='none')
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                f'{v*100:.1f}%', va='center', fontsize=9, color='#a1a1aa',
                                fontfamily='JetBrains Mono')
                    ax.set_xlim(0, 1.1)
                    ax.set_xlabel('')
                    ax.set_title('')
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # Feedback
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("**Feedback** — Was this prediction correct?")
                fb_cols = st.columns(3)
                with fb_cols[0]:
                    if st.button("✅ Correct", use_container_width=True, key="fb_correct"):
                        submit_feedback(pred["id"], diff)
                        st.toast("Thanks! Feedback recorded.")
                with fb_cols[1]:
                    if st.button("It's Easy", use_container_width=True, key="fb_easy"):
                        submit_feedback(pred["id"], "Easy")
                        st.toast("Feedback: Easy — recorded.")
                with fb_cols[2]:
                    if st.button("It's Hard", use_container_width=True, key="fb_hard"):
                        submit_feedback(pred["id"], "Hard")
                        st.toast("Feedback: Hard — recorded.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:3rem 2rem;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔮</div>
                <div style="color:var(--text-muted); font-size:0.9rem;">
                    Enter a question on the left and click<br>
                    <strong style="color:var(--accent-indigo);">Analyze & Predict</strong> to classify difficulty
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Bottom Section: Add Questions + History ──
    col_add, col_hist = st.columns(2, gap="large")

    with col_add:
        st.markdown('<div class="section-title">➕ Add Labeled Question</div>', unsafe_allow_html=True)
        st.markdown('<span style="color:var(--text-muted); font-size:0.8rem;">Submit verified questions to expand the training set for future retraining cycles.</span>', unsafe_allow_html=True)

        with st.form("add_q_form", clear_on_submit=True):
            nq_text = st.text_area("Question:", placeholder="Enter a new labeled question...", height=80)
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                nq_subj = st.selectbox("Subject", list(SUBJECT_TOPICS.keys()), key="nq_subj")
            with fc2:
                nq_topic = st.selectbox("Topic", SUBJECT_TOPICS[nq_subj], key="nq_topic")
            with fc3:
                nq_diff = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="nq_diff")
            submitted = st.form_submit_button("💾 Save to Database", use_container_width=True)
            if submitted:
                if not nq_text.strip():
                    st.error("Question text cannot be empty.")
                else:
                    add_custom_question(nq_text, nq_subj, nq_topic, nq_diff)
                    st.success("Question saved to SQLite! It will be included in the next retraining.")

    with col_hist:
        st.markdown('<div class="section-title">📜 Recent Predictions</div>', unsafe_allow_html=True)
        history = get_prediction_history(limit=5)
        if not history.empty:
            for _, r in history.iterrows():
                diff_tag = r['predicted_difficulty']
                badge_cls = f"badge-{diff_tag.lower()}" if diff_tag else "tag"
                q_preview = r['question_text'][:90] + ("..." if len(r['question_text']) > 90 else "")
                actual = r['actual_difficulty'] or "—"

                st.markdown(f"""
                <div class="history-item">
                    <div class="history-q">"{q_preview}"</div>
                    <div class="history-meta">
                        <span class="tag">{r['subject']}</span>
                        <span class="tag">{r['topic']}</span>
                        <span class="{badge_cls}" style="padding:0.15rem 0.6rem; font-size:0.7rem;">{diff_tag}</span>
                        <span style="color:var(--text-muted); font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
                            {r['confidence']*100:.0f}% · Actual: {actual}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:2rem;">
                <span style="color:var(--text-muted); font-size:0.85rem;">No predictions logged yet</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""<div class="footer">
        DifficultyAI © 2025 · Powered by XGBoost + NLTK + Streamlit · Production ML Pipeline
    </div>""", unsafe_allow_html=True)
