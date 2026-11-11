import streamlit as st
import requests

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Financial Analyst",
    page_icon="📊",
    layout="wide"
)

# ── Backend URL ────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1B3A6B;
    text-align: center;
    padding: 1rem 0;
}
.subtitle {
    text-align: center;
    color: #6B7280;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.answer-box {
    background: #f0fff4;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #16A34A;
    margin: 1rem 0;
}
.source-box {
    background: #fefce8;
    padding: 0.8rem;
    border-radius: 6px;
    border-left: 3px solid #F59E0B;
    margin: 0.3rem 0;
    font-size: 0.85rem;
}
.metric-card {
    background: #f0f7ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2563EB;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">📊 Visual Financial Analyst</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-Powered Financial Document Analysis using Multimodal RAG</div>',
    unsafe_allow_html=True
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    st.markdown("Upload any financial PDF — Annual Report, Balance Sheet, etc.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload financial PDFs like annual reports"
    )

    if uploaded_file is not None:
        st.success(f"📄 {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")

        if st.button(
            "🚀 Process Document",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("📖 Reading PDF and creating embeddings... (2-3 mins)"):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/api/upload",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.doc_loaded = True
                        st.session_state.doc_name = result["doc_name"]
                        st.session_state.doc_stats = result
                        st.session_state.chat_history = []
                        st.session_state.current_question = ""
                        st.success("✅ Document Ready!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")

                except requests.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")

    st.divider()

    # ── Quick connect to already processed document ────────────────────────
    st.subheader("⚡ Already Processed?")
    existing_doc = st.text_input(
        "Enter document name:",
        placeholder="TCS-annual report 2025-2026"
    )

    if st.button("🔗 Connect", use_container_width=True):
        if existing_doc.strip():
            st.session_state.doc_loaded = True
            st.session_state.doc_name = existing_doc.strip()
            st.session_state.doc_stats = {
                "pages_processed": 333,
                "chunks_created": 679,
                "vectors_stored": 679
            }
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.success(f"✅ Connected to {existing_doc.strip()}!")
            st.rerun()
        else:
            st.warning("Please enter a document name.")

    st.divider()

    # Show stats after upload or connection
    if st.session_state.doc_loaded:
        st.subheader("📈 Document Stats")
        stats = st.session_state.doc_stats
        st.metric("Pages Processed", stats.get("pages_processed", 0))
        st.metric("Text Chunks", stats.get("chunks_created", 0))
        st.metric("Vectors Stored", stats.get("vectors_stored", 0))

        st.divider()

        if st.button("🗑️ Upload New Document", use_container_width=True):
            st.session_state.doc_loaded = False
            st.session_state.doc_name = ""
            st.session_state.doc_stats = {}
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()

# ── Main Area ──────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.markdown("### 👋 Welcome to Visual Financial Analyst")
    st.markdown("Upload a financial PDF on the left to get started.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\n📂 Upload a financial PDF from the sidebar")
    with col2:
        st.info("**Step 2**\n\n🤖 AI reads and understands the entire document")
    with col3:
        st.info("**Step 3**\n\n💬 Ask any financial question and get answers")

    st.divider()
    st.markdown("#### 💡 Example Questions:")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("• What is the total revenue?")
        st.markdown("• What was the net profit?")
        st.markdown("• What are the key risk factors?")
    with col2:
        st.markdown("• What is the earnings per share?")
        st.markdown("• How did revenue grow this year?")
        st.markdown("• What is the debt to equity ratio?")

else:
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Document Info"])

    # ── Tab 1: Q&A ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("💬 Ask Financial Questions")
        st.markdown("**Quick Questions:**")

        col1, col2, col3, col4 = st.columns(4)

        if col1.button("💰 Total Revenue?", use_container_width=True):
            st.session_state.current_question = "What is the total revenue?"
        if col2.button("📈 Net Profit?", use_container_width=True):
            st.session_state.current_question = "What was the net profit?"
        if col3.button("⚠️ Key Risks?", use_container_width=True):
            st.session_state.current_question = "What are the key risk factors?"
        if col4.button("📊 EPS?", use_container_width=True):
            st.session_state.current_question = "What is the earnings per share?"

        st.divider()

        question = st.text_input(
            "Or type your own question:",
            value=st.session_state.current_question,
            placeholder="e.g. What was the revenue growth in FY2025?"
        )

        col_ask, col_clear = st.columns([3, 1])
        ask_btn = col_ask.button(
            "🔍 Ask",
            type="primary",
            use_container_width=True
        )
        clear_btn = col_clear.button(
            "🗑️ Clear",
            use_container_width=True
        )

        if clear_btn:
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.rerun()

        if ask_btn and question.strip():
            with st.spinner("🤖 Analyzing document..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/api/query",
                        json={
                            "question": question.strip(),
                            "doc_name": st.session_state.doc_name
                        },
                        timeout=60
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.chat_history.insert(0, {
                            "question": question.strip(),
                            "answer": result.get("answer", ""),
                            "confidence": result.get("confidence", 0.0),
                            "sources": result.get("sources", [])
                        })
                        st.session_state.current_question = ""
                    else:
                        st.error(f"Query failed: {response.text}")

                except requests.RequestException as e:
                    st.error(f"Error connecting to backend: {e}")

        if st.session_state.chat_history:
            st.divider()
            st.markdown("### 📝 Answers")

            for i, item in enumerate(st.session_state.chat_history):
                num = len(st.session_state.chat_history) - i
                st.markdown(f"**Q{num}: {item['question']}**")

                st.markdown(
                    f'<div class="answer-box">{item["answer"]}</div>',
                    unsafe_allow_html=True
                )

                conf = item["confidence"]
                if conf > 0.7:
                    color = "green"
                elif conf > 0.4:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(
                    f"<small style='color:{color}'>🎯 Confidence: {conf * 100:.0f}%</small>",
                    unsafe_allow_html=True
                )
                st.progress(float(conf))

                if item["sources"]:
                    with st.expander(
                        f"📌 View Sources ({len(item['sources'])} found)"
                    ):
                        for src in item["sources"]:
                            st.markdown(
                                f'<div class="source-box">'
                                f'📄 Page <b>{src.get("page_num", "?")}</b> | '
                                f'Section: <b>{src.get("section", "Unknown")}</b> | '
                                f'Relevance: <b>{src.get("score", 0):.0%}</b>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                st.divider()
        else:
            st.info("👆 Click a quick question or type your own above!")

    # ── Tab 2: Document Info ───────────────────────────────────────────────
    with tab2:
        st.subheader(f"📄 Document: {st.session_state.doc_name}")
        stats = st.session_state.doc_stats

        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Pages", stats.get("pages_processed", 0))
        col2.metric("🔤 Chunks", stats.get("chunks_created", 0))
        col3.metric("🧠 Vectors", stats.get("vectors_stored", 0))

        st.divider()
        st.markdown("**How it works:**")
        st.markdown("""
        1. 📄 PDF was read page by page using PyMuPDF
        2. ✂️ Text was split into 400-word overlapping chunks
        3. 🔢 Each chunk was converted to a 3072-dimension vector using Gemini Embedding
        4. 💾 All vectors were stored in Qdrant local database
        5. 🔍 When you ask a question, it finds the most similar chunks
        6. 🤖 Gemini reads those chunks and generates your answer
        """)

        st.divider()
        st.success("✅ Status: Ready to answer questions")
        st.info(f"🤖 Model: {BACKEND_URL.replace('http://', '')}")