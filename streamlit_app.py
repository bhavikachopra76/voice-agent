"""
Streamlit Frontend — Voice Agent Chat UI.

Run with:
    streamlit run streamlit_app.py

(Make sure the FastAPI server is running first — see README)
"""

import streamlit as st
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

# ──────────────────────────────────────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Voice Agent — Document Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark premium aesthetic
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Typography ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero header ────────────────────────────────────────────────────── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0F172A 100%);
    }
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #7C3AED, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Source-chunk card ───────────────────────────────────────────────── */
    .src-card {
        background: rgba(124, 58, 237, 0.07);
        border: 1px solid rgba(124, 58, 237, 0.18);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 0.84rem;
        line-height: 1.55;
        color: #D1D5DB;
    }
    .sim-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7C3AED, #3B82F6);
        color: #fff;
        padding: 2px 11px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* ── Upload result ──────────────────────────────────────────────────── */
    .upload-ok {
        background: rgba(16, 185, 129, 0.09);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 8px;
        line-height: 1.7;
    }

    /* ── Status indicator ───────────────────────────────────────────────── */
    .dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot-on  { background: #10B981; box-shadow: 0 0 6px #10B981; }
    .dot-off { background: #EF4444; box-shadow: 0 0 6px #EF4444; }

    /* ── Clean-up ───────────────────────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, audio?, sources?}]
if "history" not in st.session_state:
    st.session_state.history = []           # [{role, content}]  — sent to LLM

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _server_online() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _render_sources(sources: list[dict]):
    """Show retrieved source chunks inside an expander."""
    if not sources:
        return
    with st.expander(f"📎  {len(sources)} source chunks retrieved"):
        for s in sources:
            pct = int(s["similarity"] * 100)
            text = s["content"][:400] + ("…" if len(s["content"]) > 400 else "")
            st.markdown(
                f'<div class="src-card">'
                f'<span class="sim-badge">{pct}% match</span><br>{text}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

online = _server_online()

with st.sidebar:
    st.markdown("# 🎙️ Voice Agent")
    st.caption("AI-powered document assistant with voice")

    dot = "dot-on" if online else "dot-off"
    label = "Server Online" if online else "Server Offline"
    st.markdown(f'<span class="dot {dot}"></span> {label}', unsafe_allow_html=True)

    st.divider()

    # ── PDF upload ────────────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"],
                                     help="Upload a PDF to add it to the knowledge base.")

    if uploaded_file and online:
        if st.button("⬆️  Upload & Process", use_container_width=True, type="primary"):
            with st.status("Processing document…", expanded=True) as status:
                st.write("📤 Uploading to server…")
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(),
                                  "application/pdf")}
                try:
                    resp = requests.post(f"{API_URL}/upload", files=files, timeout=300)
                    if resp.ok:
                        d = resp.json()
                        status.update(label="✅ Upload complete!", state="complete")
                        st.markdown(
                            f'<div class="upload-ok">'
                            f'<strong>{d["filename"]}</strong><br>'
                            f'📑 {d["pages_extracted"]} pages &nbsp;·&nbsp; '
                            f'🧩 {d["chunks_created"]} chunks &nbsp;·&nbsp; '
                            f'☁️ {d["chunks_uploaded"]} uploaded'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        status.update(label="Upload failed", state="error")
                        st.error(resp.json().get("detail", resp.text))
                except requests.ConnectionError:
                    status.update(label="Connection error", state="error")
                    st.error("Could not reach the server.")

    st.divider()

    # ── Document list ─────────────────────────────────────────────────────
    st.subheader("📚 Knowledge Base")
    if online:
        try:
            resp = requests.get(f"{API_URL}/documents", timeout=5)
            docs = resp.json().get("documents", []) if resp.ok else []
            if docs:
                for name in docs:
                    c1, c2 = st.columns([4, 1])
                    c1.markdown(f"📄 `{name}`")
                    if c2.button("🗑️", key=f"del_{name}", help=f"Delete {name}"):
                        requests.delete(f"{API_URL}/documents/{name}", timeout=10)
                        st.rerun()
            else:
                st.info("No documents yet — upload a PDF above!")
        except Exception:
            st.warning("Could not fetch documents.")
    else:
        st.warning("Start the server first.\n\n`uvicorn server:app --reload`")

    st.divider()

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("Built with FastAPI · Streamlit · Cerebras AI · Edge-TTS")

# ──────────────────────────────────────────────────────────────────────────────
# Main chat area
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Voice Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'Upload documents and ask questions — get instant voice-powered answers'
    '</div>',
    unsafe_allow_html=True,
)

# ── Render conversation history ──────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")
        _render_sources(msg.get("sources", []))

# ── Chat input ───────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):

    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # Generate the assistant response
    with st.chat_message("assistant", avatar="🤖"):
        if not online:
            st.error("The server is offline. Start it with `uvicorn server:app --reload`")
        else:
            with st.spinner("Thinking…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={
                            "question": prompt,
                            "history":  st.session_state.history,
                        },
                        timeout=60,
                    )

                    if resp.ok:
                        data = resp.json()

                        # Answer text
                        st.markdown(data["answer"])

                        # Audio playback (autoplay for latest response)
                        audio_bytes = None
                        try:
                            audio_resp = requests.get(
                                f"{API_URL}{data['audio_url']}", timeout=10,
                            )
                            if audio_resp.ok:
                                audio_bytes = audio_resp.content
                                st.audio(audio_bytes, format="audio/mp3",
                                         autoplay=True)
                        except Exception:
                            pass  # audio is nice-to-have, don't crash

                        # Source chunks
                        _render_sources(data.get("sources", []))

                        # Persist to session state
                        st.session_state.messages.append({
                            "role":    "assistant",
                            "content": data["answer"],
                            "audio":   audio_bytes,
                            "sources": data.get("sources", []),
                        })
                        st.session_state.history.append(
                            {"role": "user", "content": prompt})
                        st.session_state.history.append(
                            {"role": "assistant", "content": data["answer"]})
                    else:
                        detail = resp.json().get("detail", "Unknown error")
                        st.error(f"❌ {detail}")

                except requests.ConnectionError:
                    st.error("Cannot connect to the server. Is it running?")
                except requests.Timeout:
                    st.error("Request timed out — the server may be overloaded.")
