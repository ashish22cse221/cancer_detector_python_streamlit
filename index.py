"""
Thermo Spectroscope — Phase 2
Streamlit Cancer Detection App
Run: streamlit run app.py
All .pkl files must be in the same folder as this script.
"""

# ── Auto-install missing dependencies ────────────────────────────────────────
import sys, subprocess

# Maps: import_name -> pip package name
REQUIRED = {
    "streamlit":    "streamlit",
    "numpy":        "numpy",
    "joblib":       "joblib",
    "cv2":          "opencv-python",
    "matplotlib":   "matplotlib",
    "PIL":          "Pillow",
    "sklearn":      "scikit-learn",
}

_missing = []
for _import_name, _pip_name in REQUIRED.items():
    try:
        __import__(_import_name)
    except ImportError:
        _missing.append(_pip_name)

if _missing:
    print(f"\n[Thermo Spectroscope] Installing missing packages: {', '.join(_missing)}\n")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + _missing
    )
    print("\n[Thermo Spectroscope] Installation complete. Starting app...\n")

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import joblib, os, io, time, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermo Spectroscope",
    page_icon="🌡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F0F2F6;
}

.block-container {
    padding: 2.5rem 1.5rem 3rem !important;
    max-width: 780px !important;
}

* { box-sizing: border-box; }

/* ── Typography ── */
h1, h2, h3, h4, p, div, span, label, li {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Nav header ── */
.nav-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1C2B4A;
    border-radius: 12px;
    padding: 14px 24px;
    margin-bottom: 28px;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.03em;
}
.nav-brand-icon { font-size: 1.3rem; }
.nav-steps {
    display: flex;
    align-items: center;
    gap: 0;
}
.nav-step {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.35);
    padding: 4px 10px;
}
.nav-step.done  { color: #4ADE80; }
.nav-step.active { color: #FFFFFF; font-weight: 700; }
.nav-step-num {
    width: 20px; height: 20px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 700;
    background: rgba(255,255,255,0.12);
}
.nav-step.done  .nav-step-num { background: #4ADE80; color: #052E16; }
.nav-step.active .nav-step-num { background: #FFFFFF; color: #1C2B4A; }
.nav-sep { color: rgba(255,255,255,0.2); font-size: 0.7rem; }

/* ── Page header ── */
.page-header {
    margin-bottom: 24px;
}
.page-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 4px 0;
    line-height: 1.3;
}
.page-subtitle {
    font-size: 0.875rem;
    color: #64748B;
    margin: 0;
}

/* ── Card ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-title {
    font-size: 0.875rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 14px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #F1F5F9;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Stat tiles ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.stat-tile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
}
.stat-tile-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.stat-tile-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1;
}

/* ── Status badge ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.775rem;
    font-weight: 600;
}
.badge-green  { background: #DCFCE7; color: #166534; }
.badge-red    { background: #FEE2E2; color: #991B1B; }
.badge-blue   { background: #DBEAFE; color: #1D4ED8; }
.badge-amber  { background: #FEF3C7; color: #92400E; }
.badge-slate  { background: #F1F5F9; color: #475569; }

/* ── Info / warning boxes ── */
.box-info {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.825rem;
    color: #1E3A5F;
    margin-bottom: 14px;
    line-height: 1.6;
}
.box-warning {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 3px solid #F59E0B;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.80rem;
    color: #78350F;
    margin-top: 16px;
    line-height: 1.6;
}
.box-error {
    background: #FFF1F2;
    border: 1px solid #FECDD3;
    border-left: 3px solid #F43F5E;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    font-size: 0.825rem;
    color: #881337;
    margin-bottom: 14px;
    line-height: 1.6;
}

/* ── Result banners ── */
.result-positive {
    background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
    border: 1.5px solid #FDA4AF;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-negative {
    background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
    border: 1.5px solid #86EFAC;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-tag {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.result-label {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.2;
}
.result-positive .result-tag   { color: #BE123C; }
.result-positive .result-label { color: #9F1239; }
.result-negative .result-tag   { color: #15803D; }
.result-negative .result-label { color: #166534; }
.result-conf {
    text-align: right;
}
.result-conf-label {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}
.result-positive .result-conf-label { color: #BE123C; }
.result-negative .result-conf-label { color: #15803D; }
.result-conf-value {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1;
}
.result-positive .result-conf-value { color: #9F1239; }
.result-negative .result-conf-value { color: #166534; }
.result-conf-unit {
    font-size: 1rem;
    font-weight: 600;
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 20px 0 10px 0;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: #E2E8F0;
    margin: 20px 0;
}

/* ── Option card (camera / upload) ── */
.option-card {
    background: #FFFFFF;
    border: 2px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.option-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.option-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: #EFF6FF;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.option-title {
    font-size: 0.925rem;
    font-weight: 700;
    color: #0F172A;
}
.option-desc {
    font-size: 0.80rem;
    color: #64748B;
    margin-bottom: 14px;
    line-height: 1.5;
}

/* ── Streamlit button overrides ── */
.stButton > button {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    border: 1.5px solid transparent !important;
    transition: all 0.18s ease !important;
    width: 100%;
    cursor: pointer !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button[data-testid*="primary"] {
    background: #1C2B4A !important;
    color: #FFFFFF !important;
    border-color: #1C2B4A !important;
    box-shadow: 0 2px 8px rgba(28,43,74,0.25) !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid*="primary"]:hover {
    background: #263d6b !important;
    box-shadow: 0 4px 12px rgba(28,43,74,0.35) !important;
    transform: translateY(-1px) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #374151 !important;
    border-color: #D1D5DB !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
    transform: translateY(-1px) !important;
}

/* Download button */
.stDownloadButton > button {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 1.5px solid #D1D5DB !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}
.stDownloadButton > button:hover {
    background: #F9FAFB !important;
    border-color: #9CA3AF !important;
    transform: translateY(-1px) !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #1C2B4A, #3B5998) !important;
    border-radius: 4px !important;
    height: 6px !important;
}
.stProgress > div {
    background: #E2E8F0 !important;
    border-radius: 4px !important;
    height: 6px !important;
}

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #F8FAFC !important;
    border: 2px dashed #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 20px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #94A3B8 !important;
    background: #F1F5F9 !important;
}

/* Success/error/info messages */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* Image display */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    font-size: 0.85rem !important;
    color: #64748B !important;
}

/* Remove default st.columns gap issues */
[data-testid="stVerticalBlock"] > div:empty { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        svm    = joblib.load(os.path.join(BASE_DIR, "svm_cancer_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        pca    = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))
        return svm, scaler, pca, True
    except Exception as e:
        return None, None, None, str(e)

svm, scaler, pca, model_status = load_models()
MODEL_OK = model_status is True

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"page": 1, "image_pil": None, "result": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

IMG_SIZE = (64, 64)

# ── Navigation helper ─────────────────────────────────────────────────────────
def goto(n):
    st.session_state.page = n
    st.rerun()

# ── Nav header ────────────────────────────────────────────────────────────────
def render_nav(current):
    steps = ["Launch", "Overview", "Input", "Processing", "Report"]
    items = []
    for i, label in enumerate(steps, 1):
        if i < current:
            cls = "done"
            num = "✓"
        elif i == current:
            cls = "active"
            num = str(i)
        else:
            cls = ""
            num = str(i)
        items.append(
            f'<div class="nav-step {cls}">'
            f'<div class="nav-step-num">{num}</div>'
            f'{label}</div>'
        )
        if i < len(steps):
            items.append('<span class="nav-sep">›</span>')

    st.markdown(f"""
    <div class="nav-header">
        <div class="nav-brand">
            <span class="nav-brand-icon">🌡</span>
            Thermo Spectroscope
        </div>
        <div class="nav-steps">{''.join(items)}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Camera helpers ────────────────────────────────────────────────────────────
def check_camera():
    try:
        cap = cv2.VideoCapture(0)
        ok  = cap.isOpened()
        cap.release()
        return ok
    except:
        return False

def capture_frame():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Camera could not be opened."
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, "Failed to read frame from camera."
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), None
    except Exception as e:
        return None, str(e)

# ── Core analysis ─────────────────────────────────────────────────────────────
def run_analysis(img_pil):
    img_rgb = img_pil.convert("RGB").resize(IMG_SIZE)
    arr     = np.array(img_rgb, dtype=np.float32).flatten() / 255.0
    arr_sc  = scaler.transform([arr])
    arr_pca = pca.transform(arr_sc)

    pred      = int(svm.predict(arr_pca)[0])
    proba     = svm.predict_proba(arr_pca)[0]
    is_cancer = pred == 1
    conf      = round(float(proba[pred]) * 100, 1)

    # Build heatmap via PCA inverse
    inv    = pca.inverse_transform(arr_pca)
    hm     = np.abs(inv[0] - scaler.mean_).reshape(64, 64, 3).mean(axis=2)
    hm     = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    w, h   = img_pil.size
    hm_up  = cv2.resize(hm.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#FFFFFF")
    cmap = "Reds" if is_cancer else "Blues"
    orig = img_pil.convert("RGB")

    axes[0].imshow(orig)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold",
                      color="#374151", pad=10)

    axes[1].imshow(orig)
    im = axes[1].imshow(hm_up, cmap=cmap, alpha=0.52)
    axes[1].set_title("Activation Overlay", fontsize=11, fontweight="bold",
                      color="#374151", pad=10)
    cb1 = plt.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)
    cb1.ax.tick_params(labelsize=7, colors="#94A3B8")
    cb1.outline.set_edgecolor("#E2E8F0")

    axes[2].imshow(hm_up, cmap=cmap)
    axes[2].set_title("Intensity Map", fontsize=11, fontweight="bold",
                      color="#374151", pad=10)
    cb2 = plt.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(0, 1), cmap=cmap
        ),
        ax=axes[2], fraction=0.035, pad=0.02
    )
    cb2.ax.tick_params(labelsize=7, colors="#94A3B8")
    cb2.outline.set_edgecolor("#E2E8F0")

    for ax in axes:
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    title  = "Cancer Detected" if is_cancer else "No Cancer Detected"
    tcolor = "#9F1239" if is_cancer else "#166534"
    fig.suptitle(
        f"{title}  ·  Confidence: {conf}%",
        fontsize=13, fontweight="bold", color=tcolor, y=1.02
    )
    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="#FFFFFF")
    plt.close(fig)
    buf.seek(0)

    return {
        "is_cancer":      is_cancer,
        "label":          title,
        "confidence":     conf,
        "cancer_prob":    round(float(proba[1]) * 100, 1),
        "noncancer_prob": round(float(proba[0]) * 100, 1),
        "heatmap_bytes":  buf.read(),
        "analysis_id":    f"TS-{int(time.time()):X}",
    }

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — LAUNCH
# ════════════════════════════════════════════════════════════════════
if st.session_state.page == 1:

    render_nav(1)

    # Hero block
    st.markdown("""
    <div style="text-align:center; padding: 32px 0 28px;">
        <div style="font-size:3.5rem; margin-bottom:14px;">🌡</div>
        <h1 style="font-size:2rem; font-weight:800; color:#0F172A;
                   margin:0 0 8px; letter-spacing:-0.01em;">
            Thermo Spectroscope
        </h1>
        <p style="font-size:0.95rem; color:#64748B; margin:0 auto;
                  max-width:420px; line-height:1.7;">
            Multispectral thermal image analysis for cancer detection
            using a trained Support Vector Machine classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature pills row
    st.markdown("""
    <div style="display:flex; justify-content:center; gap:8px;
                flex-wrap:wrap; margin-bottom:28px;">
        <span class="badge badge-blue">SVM · RBF Kernel</span>
        <span class="badge badge-blue">PCA · 100 Components</span>
        <span class="badge badge-blue">950 MSI Training Images</span>
        <span class="badge badge-blue">78.4% Accuracy</span>
    </div>
    """, unsafe_allow_html=True)

    # Model status card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Status</div>', unsafe_allow_html=True)

    if MODEL_OK:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:4px 0;">
            <span style="font-size:0.875rem; color:#374151;">SVM Classifier</span>
            <span class="badge badge-green">● Loaded</span>
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:4px 0;">
            <span style="font-size:0.875rem; color:#374151;">StandardScaler</span>
            <span class="badge badge-green">● Loaded</span>
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:4px 0;">
            <span style="font-size:0.875rem; color:#374151;">PCA Transform</span>
            <span class="badge badge-green">● Loaded</span>
        </div>
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:8px 0 0;">
            <span style="font-size:0.875rem; color:#374151;">System Ready</span>
            <span class="badge badge-green">● All systems operational</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="box-error">
            <strong>Model files not found.</strong> Ensure the following files are in
            the same folder as <code>app.py</code>:<br><br>
            &nbsp;&nbsp;• <code>svm_cancer_model.pkl</code><br>
            &nbsp;&nbsp;• <code>scaler.pkl</code><br>
            &nbsp;&nbsp;• <code>pca.pkl</code><br><br>
            <em>Error: {model_status}</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    col_l, col_btn, col_r = st.columns([1.5, 2, 1.5])
    with col_btn:
        if st.button("Get Started →", type="primary",
                     use_container_width=True, disabled=not MODEL_OK):
            goto(2)

    st.markdown("""
    <div class="box-warning">
        <strong>⚠ Research Use Only.</strong> This tool is intended for
        research and educational purposes only. It is not a certified medical
        diagnostic device and must not be used as a substitute for professional
        clinical assessment.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == 2:

    render_nav(2)

    st.markdown("""
    <div class="page-header">
        <div class="page-title">System Overview</div>
        <div class="page-subtitle">
            Review model performance and methodology before running an analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Performance stats
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-tile">
            <div class="stat-tile-label">Test Accuracy</div>
            <div class="stat-tile-value">78.4%</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-label">ROC-AUC</div>
            <div class="stat-tile-value">0.867</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-label">Cancer Recall</div>
            <div class="stat-tile-value">89%</div>
        </div>
        <div class="stat-tile">
            <div class="stat-tile-label">CV Score</div>
            <div class="stat-tile-value">78.2%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology card
    st.markdown("""
    <div class="card">
        <div class="card-title">Methodology</div>
        <div style="font-size:0.875rem; color:#374151; line-height:1.8;">
            The classifier was trained on the <strong>Multispectral MSI Dataset</strong>
            (500 cancer + 450 non-cancer thermal images).
            <br><br>
            <strong>Pipeline:</strong> Raw pixels (64×64 RGB) are flattened and
            normalised using <strong>StandardScaler</strong>, reduced to
            <strong>100 principal components</strong> via PCA (83.6% variance retained),
            then classified by an <strong>SVM with RBF kernel</strong>
            (C=10, balanced class weights).
            <br><br>
            <strong>Heatmap generation:</strong> The activation map is produced by
            inverting the PCA transform and computing per-pixel deviation from the
            scaler mean, providing a spatial visualisation of the most influential regions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training info
    st.markdown("""
    <div class="box-info">
        <strong>Training dataset:</strong> 950 multispectral thermal images ·
        500 cancer · 450 non-cancer · Binary classification ·
        5-fold stratified cross-validation
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col_back, spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Back", use_container_width=True, key="p2_back"):
            goto(1)
    with col_next:
        if st.button("Continue →", type="primary",
                     use_container_width=True, key="p2_next"):
            goto(3)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — IMAGE INPUT
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == 3:

    render_nav(3)

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Image Input</div>
        <div class="page-subtitle">
            Provide a thermal image via live camera capture or file upload.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cam_ok = check_camera()

    # ── Option 1: Camera ──
    st.markdown("""
    <div class="option-card">
        <div class="option-card-header">
            <div class="option-icon">📷</div>
            <div class="option-title">Option 1 — Live Camera Capture</div>
        </div>
        <div class="option-desc">
            Capture a frame directly from a connected thermal or standard camera.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if cam_ok:
        st.markdown("""
        <span class="badge badge-green" style="margin-bottom:10px; display:inline-flex;">
            ● Camera connected
        </span>
        """, unsafe_allow_html=True)
        col_cap, col_sp = st.columns([1, 2])
        with col_cap:
            if st.button("Capture Frame", use_container_width=True, key="btn_capture"):
                with st.spinner("Capturing from camera..."):
                    img, err = capture_frame()
                if err:
                    st.error(f"Capture error: {err}")
                elif img is not None:
                    st.session_state.image_pil = img
                    st.success("Frame captured successfully.")
    else:
        st.markdown("""
        <span class="badge badge-red" style="margin-bottom:8px; display:inline-flex;">
            ● No camera detected
        </span>
        <p style="font-size:0.80rem; color:#94A3B8; margin-top:6px;">
            Connect a camera and restart the application to enable this option.
        </p>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Option 2: Upload ──
    st.markdown("""
    <div class="option-card">
        <div class="option-card-header">
            <div class="option-icon">📁</div>
            <div class="option-title">Option 2 — Upload Image File</div>
        </div>
        <div class="option-desc">
            Upload a JPEG, PNG, or BMP thermal image from your device.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop or click to browse (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        key="file_uploader",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        try:
            st.session_state.image_pil = Image.open(uploaded)
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.session_state.image_pil = None

    # ── Preview ──
    if st.session_state.image_pil is not None:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)

        col_img, col_meta = st.columns([3, 2])
        with col_img:
            st.image(st.session_state.image_pil,
                     caption="Selected image",
                     use_container_width=True)
        with col_meta:
            w, h = st.session_state.image_pil.size
            mode = st.session_state.image_pil.mode
            st.markdown(f"""
            <div style="padding-top:6px;">
                <div style="margin-bottom:12px;">
                    <div style="font-size:0.68rem; font-weight:700; color:#94A3B8;
                                text-transform:uppercase; letter-spacing:0.08em;
                                margin-bottom:3px;">Dimensions</div>
                    <div style="font-size:0.925rem; font-weight:600; color:#0F172A;">
                        {w} × {h} px
                    </div>
                </div>
                <div style="margin-bottom:12px;">
                    <div style="font-size:0.68rem; font-weight:700; color:#94A3B8;
                                text-transform:uppercase; letter-spacing:0.08em;
                                margin-bottom:3px;">Colour Mode</div>
                    <div style="font-size:0.925rem; font-weight:600; color:#0F172A;">
                        {mode}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem; font-weight:700; color:#94A3B8;
                                text-transform:uppercase; letter-spacing:0.08em;
                                margin-bottom:5px;">Status</div>
                    <span class="badge badge-green">● Ready for analysis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    ready = st.session_state.image_pil is not None
    col_back, spacer, col_next = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Back", use_container_width=True, key="p3_back"):
            st.session_state.image_pil = None
            goto(2)
    with col_next:
        if st.button("Run Analysis →", type="primary",
                     use_container_width=True, disabled=not ready, key="p3_next"):
            goto(4)

    if not ready:
        st.markdown("""
        <p style="text-align:right; font-size:0.775rem; color:#94A3B8; margin-top:8px;">
            Please provide an image to continue.
        </p>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — PROCESSING
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == 4:

    if st.session_state.image_pil is None:
        goto(3)

    render_nav(4)

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Processing</div>
        <div class="page-subtitle">Analysing image — please do not navigate away.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text  = st.empty()
    log_box      = st.empty()

    steps = [
        (1, "Loading image data"),
        (2, "Normalising pixel values"),
        (3, "Applying PCA dimensionality reduction"),
        (4, "Running SVM classification"),
        (5, "Generating activation heatmap"),
        (6, "Compiling report"),
    ]

    log_lines = []

    def render_log(lines, final=False):
        rows = []
        for line in lines:
            is_done = line.startswith("✓")
            color = "#166534" if is_done else "#374151"
            rows.append(
                f'<div style="font-family:monospace; font-size:0.82rem; '
                f'color:{color}; padding:3px 0; line-height:1.5;">{line}</div>'
            )
        log_box.markdown(
            f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; '
            f'border-radius:8px; padding:16px 20px;">{"".join(rows)}</div>',
            unsafe_allow_html=True
        )

    # Animate steps 1–4
    for num, label in steps[:4]:
        pct = int(num / 6 * 100)
        status_text.markdown(
            f'<p style="font-size:0.82rem; color:#64748B; margin:0 0 8px;">'
            f'Step {num} of 6 — {label}…</p>',
            unsafe_allow_html=True
        )
        log_lines.append(f"  [{num}/6]  {label}…")
        render_log(log_lines)
        progress_bar.progress(pct)
        time.sleep(0.28)

    # Run analysis
    try:
        result = run_analysis(st.session_state.image_pil)
    except Exception as e:
        st.markdown("</div>", unsafe_allow_html=True)
        st.error(f"Analysis failed: {e}")
        col_b, _ = st.columns([1, 3])
        with col_b:
            if st.button("← Go Back", use_container_width=True):
                goto(3)
        st.stop()

    # Animate steps 5–6
    for num, label in steps[4:]:
        pct = int(num / 6 * 100)
        status_text.markdown(
            f'<p style="font-size:0.82rem; color:#64748B; margin:0 0 8px;">'
            f'Step {num} of 6 — {label}…</p>',
            unsafe_allow_html=True
        )
        log_lines.append(f"  [{num}/6]  {label}…")
        render_log(log_lines)
        progress_bar.progress(pct)
        time.sleep(0.25)

    # Done
    progress_bar.progress(100)
    log_lines.append("✓  Analysis complete.")
    render_log(log_lines, final=True)
    status_text.markdown(
        '<p style="font-size:0.82rem; color:#166534; font-weight:600; margin:0 0 8px;">'
        '✓ Complete</p>',
        unsafe_allow_html=True
    )

    st.session_state.result = result
    st.markdown("</div>", unsafe_allow_html=True)
    time.sleep(0.45)
    goto(5)


# ════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORT
# ════════════════════════════════════════════════════════════════════
elif st.session_state.page == 5:

    if st.session_state.result is None:
        goto(3)

    render_nav(5)

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Analysis Report</div>
        <div class="page-subtitle">
            SVM classification result with thermal activation heatmap.
        </div>
    </div>
    """, unsafe_allow_html=True)

    r         = st.session_state.result
    is_cancer = r["is_cancer"]

    # ── Result banner ──
    if is_cancer:
        st.markdown(f"""
        <div class="result-positive">
            <div>
                <div class="result-tag">Classification Result</div>
                <div class="result-label">⚠&nbsp; Cancer Detected</div>
                <div style="margin-top:8px;">
                    <span class="badge badge-red">High Priority</span>
                </div>
            </div>
            <div class="result-conf">
                <div class="result-conf-label">Confidence</div>
                <div class="result-conf-value">
                    {r['confidence']}<span class="result-conf-unit">%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div>
                <div class="result-tag">Classification Result</div>
                <div class="result-label">✓&nbsp; No Cancer Detected</div>
                <div style="margin-top:8px;">
                    <span class="badge badge-green">Clear</span>
                </div>
            </div>
            <div class="result-conf">
                <div class="result-conf-label">Confidence</div>
                <div class="result-conf-value">
                    {r['confidence']}<span class="result-conf-unit">%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats row ──
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, "Cancer Prob.",     f"{r['cancer_prob']}%"),
        (c2, "Non-Cancer Prob.", f"{r['noncancer_prob']}%"),
        (c3, "Classifier",       "SVM / RBF"),
        (c4, "Analysis ID",      r["analysis_id"]),
    ]
    for col, lbl, val in stats:
        with col:
            st.markdown(f"""
            <div class="stat-tile" style="margin-bottom:16px;">
                <div class="stat-tile-label">{lbl}</div>
                <div class="stat-tile-value"
                     style="font-size:{'0.85rem' if len(val) > 6 else '1.1rem'};">
                    {val}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Heatmap section ──
    st.markdown('<div class="section-label">Thermal Activation Heatmap</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="card" style="padding:16px;">', unsafe_allow_html=True)
    st.image(r["heatmap_bytes"], use_container_width=True)
    st.markdown("""
    <p style="font-size:0.775rem; color:#94A3B8; margin-top:10px; text-align:center;">
        Left: Original image &nbsp;·&nbsp;
        Centre: Activation overlay (warmer = higher influence) &nbsp;·&nbsp;
        Right: Raw intensity map
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Action buttons ──
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("New Scan", type="primary",
                     use_container_width=True, key="p5_newscan"):
            st.session_state.image_pil = None
            st.session_state.result    = None
            goto(3)
    with b2:
        if st.button("← Home", use_container_width=True, key="p5_home"):
            st.session_state.image_pil = None
            st.session_state.result    = None
            goto(1)
    with b3:
        st.download_button(
            label="⬇ Download Report",
            data=r["heatmap_bytes"],
            file_name=f"thermospectroscope_{r['analysis_id']}.png",
            mime="image/png",
            use_container_width=True,
            key="p5_download",
        )

    st.markdown("""
    <div class="box-warning">
        <strong>⚠ Research Use Only.</strong>
        This result is generated by an AI model and is intended for research
        and educational purposes only. It is not a substitute for clinical
        medical diagnosis. Always consult a qualified medical professional.
    </div>
    """, unsafe_allow_html=True)