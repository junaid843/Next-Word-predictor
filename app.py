import streamlit as st
import numpy as np
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TextGen AI",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:ital,wght@0,400;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4d9 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #1a0533 0%, #0a0a0f 60%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 720px !important;
    padding: 3rem 2rem 4rem !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 0 2.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #b87fff;
    border: 1px solid #3d1f6e;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
    background: rgba(100,40,180,0.08);
}
.hero h1 {
    font-size: clamp(2.8rem, 7vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #b87fff 50%, #7c3aff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
}
.hero p {
    font-size: 0.95rem;
    color: #6b6880;
    font-family: 'DM Mono', monospace;
    font-style: italic;
    letter-spacing: 0.02em;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2d1f4e 30%, #4a2d80 50%, #2d1f4e 70%, transparent);
    margin: 2rem 0;
}

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin: 1.2rem 0;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(184,127,255,0.4), transparent);
}

/* ── Streamlit widget overrides ── */
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #7a6fa0 !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4d9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c3aff !important;
    box-shadow: 0 0 0 3px rgba(124,58,255,0.15) !important;
    outline: none !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: #2d1f4e !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #7c3aff, #b87fff) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e4d9 !important;
}

/* Button */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #5b21b6, #7c3aff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 24px rgba(124,58,255,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #6d28d9, #8b5cf6) !important;
    box-shadow: 0 6px 32px rgba(124,58,255,0.5) !important;
    transform: translateY(-1px) !important;
}

/* Output box */
.output-box {
    background: rgba(124,58,255,0.06);
    border: 1px solid rgba(124,58,255,0.25);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    position: relative;
}
.output-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7c3aff;
    margin-bottom: 0.8rem;
}
.output-text {
    font-family: 'DM Mono', monospace;
    font-size: 1.05rem;
    line-height: 1.75;
    color: #e8e4d9;
    word-break: break-word;
}
.output-seed {
    color: #b87fff;
    font-weight: 400;
}
.output-generated {
    color: #e8e4d9;
    opacity: 0.9;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.stat-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #6b6880;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 100px;
    padding: 0.25rem 0.75rem;
    letter-spacing: 0.1em;
}

/* Error */
.error-box {
    background: rgba(255,60,60,0.06);
    border: 1px solid rgba(255,60,60,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #ff8080;
    margin-top: 1rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(124,58,255,0.3) !important;
    border-radius: 12px !important;
    background: rgba(124,58,255,0.04) !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label {
    color: #7a6fa0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* Success */
[data-testid="stAlert"] {
    background: rgba(40,200,100,0.07) !important;
    border: 1px solid rgba(40,200,100,0.2) !important;
    border-radius: 12px !important;
    color: #7effa0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Columns gap */
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loader ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_files(model_file, tokenizer_file, maxlen_file):
    import tensorflow as tf
    model = tf.keras.models.load_model(model_file)
    tokenizer = pickle.load(tokenizer_file)
    max_seq_len = pickle.load(maxlen_file)
    return model, tokenizer, max_seq_len


# ─── Text Generator ──────────────────────────────────────────────────────────
def generate_text(seed_text, next_words, model, tokenizer, max_seq_len, temperature=1.0):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    result = seed_text.strip()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([result])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding="pre")
        predictions = model.predict(token_list, verbose=0)[0]

        # Temperature sampling
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        predicted_index = np.random.choice(len(predictions), p=predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        if output_word:
            result += " " + output_word
    return result


# ─── UI ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ LSTM · Next Word Prediction</div>
    <h1>TextGen<br/>Studio</h1>
    <p>load your model · type a seed · watch it write</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Model Upload ──
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.15em;
text-transform:uppercase;color:#7a6fa0;margin-bottom:1rem;">① Upload Model Files</p>""",
unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    model_file = st.file_uploader("Model (.h5)", type=["h5"], label_visibility="visible")
with col2:
    tokenizer_file = st.file_uploader("Tokenizer (.pkl)", type=["pkl"], label_visibility="visible")
with col3:
    maxlen_file = st.file_uploader("Maxlen (.pkl)", type=["pkl"], label_visibility="visible")

st.markdown('</div>', unsafe_allow_html=True)

# ── Load button + state ──
model, tokenizer, max_seq_len = None, None, None

if model_file and tokenizer_file and maxlen_file:
    try:
        with st.spinner("Loading model…"):
            model, tokenizer, max_seq_len = load_model_files(
                model_file, tokenizer_file, maxlen_file
            )
        st.success(f"✦ Model loaded — vocab size: {len(tokenizer.word_index):,}  ·  seq length: {max_seq_len}")
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠ Failed to load model: {e}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Generation Controls ──
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.15em;
text-transform:uppercase;color:#7a6fa0;margin-bottom:1rem;">② Configure Generation</p>""",
unsafe_allow_html=True)

seed_text = st.text_input(
    "Seed text",
    placeholder="e.g.  the history of science",
    help="Model will continue from this text"
)

col_a, col_b = st.columns(2)
with col_a:
    next_words = st.slider("Words to generate", min_value=5, max_value=100, value=20, step=5)
with col_b:
    temperature = st.select_slider(
        "Creativity (temperature)",
        options=[0.3, 0.5, 0.7, 1.0, 1.2, 1.5],
        value=1.0,
        help="Low = predictable, High = creative"
    )

generate_btn = st.button("✦  Generate Text", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Output ──
if generate_btn:
    if model is None:
        st.markdown('<div class="error-box">⚠ Upload all three model files first.</div>', unsafe_allow_html=True)
    elif not seed_text.strip():
        st.markdown('<div class="error-box">⚠ Enter a seed text to start generation.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Generating…"):
            try:
                result = generate_text(
                    seed_text, next_words, model, tokenizer, max_seq_len, temperature
                )
                generated_part = result[len(seed_text):].strip()

                st.markdown(f"""
                <div class="output-box">
                    <div class="output-label">✦ Generated Output</div>
                    <div class="output-text">
                        <span class="output-seed">{seed_text}</span>
                        <span class="output-generated"> {generated_part}</span>
                    </div>
                    <div class="stats-row">
                        <span class="stat-pill">words: {next_words}</span>
                        <span class="stat-pill">temp: {temperature}</span>
                        <span class="stat-pill">total chars: {len(result)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="↓  Download output as .txt",
                    data=result,
                    file_name="generated_text.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.markdown(f'<div class="error-box">⚠ Generation error: {e}</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center;margin-top:4rem;font-family:'DM Mono',monospace;
font-size:0.62rem;letter-spacing:0.15em;color:#2d2840;text-transform:uppercase;">
    LSTM · TensorFlow · Streamlit &nbsp;✦&nbsp; TextGen Studio
</div>
""", unsafe_allow_html=True)
