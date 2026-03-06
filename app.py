Content is user-generated and unverified.
1
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI · Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0b0f1a;
    --card:      #111827;
    --border:    #1f2d45;
    --accent1:   #00e5ff;
    --accent2:   #ff4081;
    --accent3:   #69ff47;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main area ── */
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Hero title ── */
.hero {
    background: linear-gradient(135deg, #0b0f1a 0%, #0f1e35 50%, #0b0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,229,255,.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(255,64,129,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    letter-spacing: .18em;
    color: var(--accent1);
    text-transform: uppercase;
    margin-bottom: .6rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 .6rem;
    background: linear-gradient(90deg, #fff 0%, var(--accent1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, border-color .2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent1); }
.metric-card .m-label {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .4rem;
}
.metric-card .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.metric-card .m-sub { font-size: .78rem; color: var(--muted); margin-top: .3rem; }
.c1 .m-value { color: var(--accent1); }
.c2 .m-value { color: var(--accent2); }
.c3 .m-value { color: var(--accent3); }
.c4 .m-value { color: #ffd740; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: .8rem;
    margin: 2rem 0 1.2rem;
}
.section-header .sh-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
.section-header h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--text); margin: 0;
    white-space: nowrap;
}
.section-header .sh-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent1);
    flex-shrink: 0;
}

/* ── Tab override ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .8rem !important;
    letter-spacing: .05em !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    padding: .5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent1) !important;
    color: #0b0f1a !important;
}

/* ── Sliders & inputs ── */
.stSlider > div > div > div > div { background: var(--accent1) !important; }
.stSlider [data-testid="stThumbValue"] { color: var(--accent1) !important; }
.stSelectbox > div > div { background: var(--card) !important; border-color: var(--border) !important; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #0084ff) !important;
    color: #0b0f1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .75rem 2rem !important;
    width: 100% !important;
    letter-spacing: .05em !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .88 !important; }

/* ── Result boxes ── */
.result-positive {
    background: linear-gradient(135deg, rgba(255,64,129,.15), rgba(255,64,129,.05));
    border: 1px solid var(--accent2);
    border-radius: 16px; padding: 1.6rem 2rem;
    text-align: center;
}
.result-negative {
    background: linear-gradient(135deg, rgba(105,255,71,.12), rgba(105,255,71,.04));
    border: 1px solid var(--accent3);
    border-radius: 16px; padding: 1.6rem 2rem;
    text-align: center;
}
.result-positive h2, .result-negative h2 {
    font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; margin: .4rem 0;
}
.result-positive h2 { color: var(--accent2); }
.result-negative h2 { color: var(--accent3); }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 12px !important; }

/* ── Info/warning blocks ── */
.stAlert { background: var(--card) !important; border-color: var(--border) !important; }

/* Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#1f2d45",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1f2d45",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

ACCENT1 = "#00e5ff"
ACCENT2 = "#ff4081"
ACCENT3 = "#69ff47"
GOLD    = "#ffd740"
PALETTE = [ACCENT1, ACCENT2, ACCENT3, GOLD, "#c77dff", "#ff9a3c"]

# ─────────────────────────────────────────────
# DATA + MODELS  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        # fallback synthetic data that matches the Pima schema
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies": np.random.randint(0,18,n),
            "Glucose": np.random.randint(60,200,n),
            "BloodPressure": np.random.randint(40,110,n),
            "SkinThickness": np.random.randint(0,60,n),
            "Insulin": np.random.randint(0,850,n),
            "BMI": np.round(np.random.uniform(18,60,n),1),
            "DiabetesPedigreeFunction": np.round(np.random.uniform(.07,2.5,n),3),
            "Age": np.random.randint(21,82,n),
            "Outcome": np.random.randint(0,2,n),
        })

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    log_m  = LogisticRegression(max_iter=1000)
    dt_m   = DecisionTreeClassifier(max_depth=3, random_state=42)
    rf_m   = RandomForestClassifier(n_estimators=100, random_state=42)

    log_m.fit(X_train_s, y_train);  log_pred  = log_m.predict(X_test_s)
    dt_m.fit(X_train_s, y_train);   dt_pred   = dt_m.predict(X_test_s)
    rf_m.fit(X_train_s, y_train);   rf_pred   = rf_m.predict(X_test_s)

    results = {
        "Logistic Regression": {
            "model": log_m, "pred": log_pred,
            "acc": accuracy_score(y_test, log_pred),
            "cm": confusion_matrix(y_test, log_pred),
            "color": ACCENT1,
        },
        "Decision Tree": {
            "model": dt_m, "pred": dt_pred,
            "acc": accuracy_score(y_test, dt_pred),
            "cm": confusion_matrix(y_test, dt_pred),
            "color": ACCENT2,
        },
        "Random Forest": {
            "model": rf_m, "pred": rf_pred,
            "acc": accuracy_score(y_test, rf_pred),
            "cm": confusion_matrix(y_test, rf_pred),
            "color": ACCENT3,
        },
    }

    return df, X, y, X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler, results

df, X, y, X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler, results = load_and_train()
feature_names = list(X.columns)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem">
        <div style="font-size:2.4rem">🩺</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(90deg,#00e5ff,#ff4081);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            DiabetesAI
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:.65rem;
                    letter-spacing:.15em;color:#64748b;text-transform:uppercase;margin-top:.2rem">
            Prediction Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    nav = st.radio(
        "Navigation",
        ["🏠  Overview", "🔬  Model Analysis", "🎯  Live Predictor", "📊  Data Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:#64748b;line-height:1.8">
    <div style="color:#00e5ff;font-weight:600;margin-bottom:.5rem">▸ DATASET</div>
    Pima Indians Diabetes<br>768 records · 8 features<br>
    <br>
    <div style="color:#00e5ff;font-weight:600;margin-bottom:.5rem">▸ MODELS</div>
    Logistic Regression<br>Decision Tree (d=3)<br>Random Forest (100 trees)
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">▸ Machine Learning · Healthcare Analytics</div>
    <h1>Diabetes Prediction<br>Dashboard</h1>
    <p>Real-time ML models trained on the Pima Indians dataset — compare algorithms,<br>
       explore distributions, and predict diabetes risk instantly.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP METRIC CARDS
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["acc"])
best_acc  = results[best_name]["acc"]
pos_rate  = y.mean()

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card c1">
    <div class="m-label">Best Accuracy</div>
    <div class="m-value">{best_acc:.1%}</div>
    <div class="m-sub">{best_name}</div>
  </div>
  <div class="metric-card c4">
    <div class="m-label">Total Samples</div>
    <div class="m-value">{len(df)}</div>
    <div class="m-sub">768 records</div>
  </div>
  <div class="metric-card c2">
    <div class="m-label">Diabetic Rate</div>
    <div class="m-value">{pos_rate:.1%}</div>
    <div class="m-sub">{int(y.sum())} positive cases</div>
  </div>
  <div class="metric-card c3">
    <div class="m-label">Features</div>
    <div class="m-value">8</div>
    <div class="m-sub">clinical variables</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
if nav == "🏠  Overview":

    def section(title):
        st.markdown(f"""
        <div class="section-header">
          <div class="sh-dot"></div>
          <h2>{title}</h2>
          <div class="sh-line"></div>
        </div>""", unsafe_allow_html=True)

    section("Model Accuracy Comparison")

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        names   = list(results.keys())
        accs    = [results[n]["acc"] for n in names]
        colors  = [results[n]["color"] for n in names]
        bars = ax.barh(names, accs, color=colors, height=0.45,
                       edgecolor="none")
        for bar, acc, col in zip(bars, accs, colors):
            ax.text(acc - 0.002, bar.get_y() + bar.get_height()/2,
                    f"{acc:.1%}", va="center", ha="right",
                    fontsize=10, fontweight="bold", color="#0b0f1a")
        ax.set_xlim(0.65, 0.82)
        ax.set_xlabel("Accuracy", fontsize=9)
        ax.axvline(sum(accs)/len(accs), color="#ffffff22", linestyle="--", linewidth=1)
        ax.set_title("Classifier Performance", fontsize=11, pad=12)
        ax.grid(axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        # Radar / polar chart
        categories = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")

        for name, col in zip(names, colors):
            from sklearn.metrics import precision_score, recall_score, f1_score
            p = results[name]
            cm_ = p["cm"]
            tn, fp, fn, tp = cm_.ravel()
            vals = [
                p["acc"],
                precision_score(y_test, p["pred"], zero_division=0),
                recall_score(y_test, p["pred"], zero_division=0),
                f1_score(y_test, p["pred"], zero_division=0),
                tn / (tn + fp) if (tn + fp) > 0 else 0,
            ]
            vals += vals[:1]
            ax.plot(angles, vals, color=col, linewidth=1.8, linestyle="solid")
            ax.fill(angles, vals, color=col, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=8, color="#94a3b8")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%","50%","75%","100%"], size=6, color="#475569")
        ax.grid(color="#1f2d45", linewidth=0.8)
        ax.spines["polar"].set_color("#1f2d45")
        ax.set_title("Multi-Metric Radar", pad=16, fontsize=10, color="#e2e8f0")

        patches = [mpatches.Patch(color=colors[i], label=names[i]) for i in range(3)]
        ax.legend(handles=patches, loc="upper right",
                  bbox_to_anchor=(1.35, 1.15), framealpha=0,
                  fontsize=7.5, labelcolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    section("ROC Curves")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name, col in zip(names, colors):
        m = results[name]["model"]
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test_s)[:, 1]
        else:
            proba = m.decision_function(X_test_s)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, linewidth=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1], color="#334155", linestyle="--", linewidth=1)
    ax.fill_between([0,1],[0,1], alpha=0.04, color="#334155")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", pad=12)
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, col) in enumerate(zip(names, colors)):
        with cols[i]:
            cm_ = results[name]["cm"]
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            sns.heatmap(cm_, annot=True, fmt="d", ax=ax,
                        cmap=sns.dark_palette(col, as_cmap=True),
                        linewidths=2, linecolor="#0b0f1a",
                        annot_kws={"size": 14, "weight": "bold", "color": "#fff"},
                        cbar=False)
            ax.set_xticklabels(["No DM", "DM"], fontsize=9)
            ax.set_yticklabels(["No DM", "DM"], fontsize=9, rotation=0)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(name, fontsize=9, color=col, pad=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ══════════════════════════════════════════════
elif nav == "🔬  Model Analysis":

    def section(title):
        st.markdown(f"""
        <div class="section-header">
          <div class="sh-dot"></div>
          <h2>{title}</h2>
          <div class="sh-line"></div>
        </div>""", unsafe_allow_html=True)

    section("Feature Importance — Random Forest")

    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(idx))],
        height=0.55, edgecolor="none"
    )
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", fontsize=8, color="#94a3b8")
    ax.set_xlabel("Importance Score")
    ax.set_title("Random Forest Feature Importances", pad=12)
    ax.grid(axis="x")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Decision Tree Visualization")
    dt_model = results["Decision Tree"]["model"]
    fig, ax = plt.subplots(figsize=(16, 6))
    plot_tree(dt_model, feature_names=feature_names,
              class_names=["No DM", "DM"],
              filled=True, ax=ax, fontsize=7,
              impurity=True, rounded=True)
    # recolor nodes
    for artist in ax.get_children():
        if hasattr(artist, "get_facecolor"):
            fc = artist.get_facecolor()
            if fc is not None:
                try:
                    r, g, b, a = fc
                    if r > g and r > b:
                        artist.set_facecolor(f"#{int(r*180+50):02x}2050")
                    elif b > r and b > g:
                        artist.set_facecolor(f"#0a{int(b*150+50):02x}{int(b*200+50):02x}")
                except Exception:
                    pass
    ax.set_title("Decision Tree (max_depth=3)", pad=14, fontsize=11)
    fig.patch.set_facecolor("#111827")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Logistic Regression · Glucose S-Curve")

    X_glu = df[["Glucose"]].values
    y_out = df["Outcome"].values
    lr_glu = LogisticRegression(max_iter=1000)
    lr_glu.fit(X_glu, y_out)
    x_lin = np.linspace(df["Glucose"].min(), df["Glucose"].max(), 400)
    y_prob = lr_glu.predict_proba(x_lin.reshape(-1,1))[:,1]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.scatter(df[y_out==0]["Glucose"], y_out[y_out==0],
               color=ACCENT1, alpha=0.25, s=12, label="No Diabetes", zorder=2)
    ax.scatter(df[y_out==1]["Glucose"], y_out[y_out==1],
               color=ACCENT2, alpha=0.35, s=12, label="Diabetes", zorder=2)
    ax.plot(x_lin, y_prob, color=GOLD, linewidth=2.5, label="P(Diabetes | Glucose)", zorder=3)
    ax.axhline(0.5, color="#334155", linestyle="--", linewidth=1, alpha=0.8)
    ax.fill_between(x_lin, y_prob, 0.5,
                    where=(y_prob > 0.5), alpha=0.08, color=ACCENT2)
    ax.fill_between(x_lin, y_prob, 0.5,
                    where=(y_prob <= 0.5), alpha=0.08, color=ACCENT1)
    ax.set_xlabel("Glucose Level (mg/dL)")
    ax.set_ylabel("P(Diabetes)")
    ax.set_title("Logistic Regression S-Curve — Glucose", pad=12)
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Classification Report")
    sel_model = st.selectbox("Select model", list(results.keys()))
    from sklearn.metrics import classification_report as cr
    report_dict = cr(y_test, results[sel_model]["pred"],
                     target_names=["No Diabetes","Diabetes"],
                     output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(
        report_df.style
            .background_gradient(cmap="Blues", subset=["precision","recall","f1-score"])
            .format(precision=3),
        use_container_width=True
    )

# ══════════════════════════════════════════════
# TAB 3 — LIVE PREDICTOR
# ══════════════════════════════════════════════
elif nav == "🎯  Live Predictor":

    st.markdown("""
    <div class="section-header">
      <div class="sh-dot"></div>
      <h2>Enter Patient Data</h2>
      <div class="sh-line"></div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies  = st.slider("Pregnancies",       0, 17,  3)
        glucose      = st.slider("Glucose (mg/dL)",  44, 199, 117)
        blood_press  = st.slider("Blood Pressure",   24, 122,  72)
    with col2:
        skin_thick   = st.slider("Skin Thickness",    0, 99,   23)
        insulin      = st.slider("Insulin (μU/mL)",   0, 846, 79)
        bmi          = st.slider("BMI",              18.0, 67.0, 32.0)
    with col3:
        dpf          = st.slider("Diabetes Pedigree", 0.08, 2.42, 0.47)
        age          = st.slider("Age",              21, 81,  33)

    st.markdown("<br>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose prediction model",
        ["Logistic Regression", "Decision Tree", "Random Forest"],
        index=2,
    )

    if st.button("🔍  Run Prediction"):
        input_arr = np.array([[pregnancies, glucose, blood_press,
                               skin_thick, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_arr)

        chosen = results[model_choice]["model"]
        pred   = chosen.predict(input_scaled)[0]
        prob   = chosen.predict_proba(input_scaled)[0]

        c_left, c_right = st.columns([1, 1.5])

        with c_left:
            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div style="font-size:2.4rem">⚠️</div>
                    <h2>HIGH RISK</h2>
                    <div style="font-family:'DM Mono',monospace;font-size:.8rem;
                                color:#64748b;margin:.4rem 0 .8rem">
                        Diabetes Detected
                    </div>
                    <div style="font-size:2.6rem;font-weight:800;
                                color:#ff4081;font-family:'Syne',sans-serif">
                        {prob[1]:.1%}
                    </div>
                    <div style="font-size:.78rem;color:#94a3b8">probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div style="font-size:2.4rem">✅</div>
                    <h2>LOW RISK</h2>
                    <div style="font-family:'DM Mono',monospace;font-size:.8rem;
                                color:#64748b;margin:.4rem 0 .8rem">
                        No Diabetes Detected
                    </div>
                    <div style="font-size:2.6rem;font-weight:800;
                                color:#69ff47;font-family:'Syne',sans-serif">
                        {prob[0]:.1%}
                    </div>
                    <div style="font-size:.78rem;color:#94a3b8">confidence</div>
                </div>""", unsafe_allow_html=True)

        with c_right:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(5, 2.2))
            ax.barh(["No Diabetes","Diabetes"], prob,
                    color=[ACCENT1, ACCENT2], height=0.4, edgecolor="none")
            for i, v in enumerate(prob):
                ax.text(v + 0.01, i, f"{v:.1%}",
                        va="center", fontsize=11, fontweight="bold",
                        color=[ACCENT1, ACCENT2][i])
            ax.set_xlim(0, 1.15)
            ax.set_title(f"Prediction Confidence · {model_choice}", pad=10, fontsize=9)
            ax.grid(axis="x")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Feature deviation from mean
            means = X.mean()
            devs  = input_arr[0] - means.values
            norm_devs = devs / X.std().values
            colors_dev = [ACCENT2 if d > 0 else ACCENT1 for d in norm_devs]

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.barh(feature_names, norm_devs, color=colors_dev,
                     height=0.5, edgecolor="none")
            ax2.axvline(0, color="#475569", linewidth=1)
            ax2.set_title("Z-Score vs Dataset Mean", pad=10, fontsize=9)
            ax2.set_xlabel("Standard Deviations from Mean", fontsize=8)
            ax2.grid(axis="x")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("""
        <div style="margin-top:1rem;padding:.8rem 1.2rem;
                    background:#111827;border:1px solid #1f2d45;
                    border-radius:10px;font-size:.75rem;
                    color:#64748b;font-family:'DM Mono',monospace">
        ⚠️ This tool is for educational purposes only. Always consult a licensed medical professional for clinical decisions.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════
elif nav == "📊  Data Explorer":

    def section(title):
        st.markdown(f"""
        <div class="section-header">
          <div class="sh-dot"></div>
          <h2>{title}</h2>
          <div class="sh-line"></div>
        </div>""", unsafe_allow_html=True)

    section("Raw Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    section("Distribution Explorer")
    feat = st.selectbox("Select feature", feature_names, index=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # KDE by class
    ax = axes[0]
    for outcome, col, label in [(0, ACCENT1, "No Diabetes"), (1, ACCENT2, "Diabetes")]:
        vals = df[df["Outcome"]==outcome][feat]
        vals.plot.kde(ax=ax, color=col, linewidth=2, label=label)
        ax.fill_between(np.linspace(vals.min(), vals.max(), 200),
                        0, ax.lines[-1].get_ydata(),
                        color=col, alpha=0.1)
    ax.set_title(f"{feat} — KDE by Outcome", pad=10)
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(True)

    # Violin
    ax2 = axes[1]
    data0 = df[df["Outcome"]==0][feat].values
    data1 = df[df["Outcome"]==1][feat].values
    vp = ax2.violinplot([data0, data1], positions=[0,1],
                        widths=0.6, showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], [ACCENT1, ACCENT2]):
        body.set_facecolor(col)
        body.set_alpha(0.5)
        body.set_edgecolor(col)
    vp["cmedians"].set_color("#fff")
    ax2.set_xticks([0,1]); ax2.set_xticklabels(["No Diabetes","Diabetes"])
    ax2.set_title(f"{feat} — Violin Plot", pad=10)
    ax2.grid(True)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Correlation Heatmap")

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(220, 20, s=90, l=40, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", ax=ax,
                cmap=cmap, linewidths=1.5, linecolor="#0b0f1a",
                annot_kws={"size": 8}, square=True, cbar_kws={"shrink": .7})
    ax.set_title("Feature Correlation Matrix", pad=14)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    section("Pairwise Scatter (Glucose vs BMI)")

    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, col, label in [(0, ACCENT1, "No DM"), (1, ACCENT2, "Diabetes")]:
        sub = df[df["Outcome"]==outcome]
        ax.scatter(sub["Glucose"], sub["BMI"],
                   color=col, alpha=0.4, s=20, label=label, edgecolors="none")
    ax.set_xlabel("Glucose (mg/dL)"); ax.set_ylabel("BMI")
    ax.set_title("Glucose vs BMI by Outcome", pad=12)
    ax.legend(framealpha=0, fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
