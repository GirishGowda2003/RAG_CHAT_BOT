import os
import re
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_groq import ChatGroq

# ── Load API key from .env file ───────────────────────────────────────────────
def load_api_key_from_env(env_path=".env"):
    api_key = ""
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                # Handles formats like: api_key : "value" or api_key=value
                match = re.search(r'api_key\s*[=:]\s*["\']?([^"\'\\n]+)["\']?', line, re.IGNORECASE)
                if match:
                    api_key = match.group(1).strip()
                    break
    return api_key

GROQ_API_KEY = load_api_key_from_env()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-PDF Chat Bot", page_icon="🤖", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "rag_chain": None,
    "messages": [],
    "pdf_processed": False,
    "groq_api_key": GROQ_API_KEY,   # Pre-filled from .env
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Cache embeddings model so it loads only once ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ── Helper: extract text ──────────────────────────────────────────────────────
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def build_rag_chain(raw_text: str, groq_api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    if not chunks:
        raise ValueError("No chunks created — check if PDFs have selectable text.")

    embeddings = load_embeddings()
    db = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})

    prompt_template = """You are a helpful assistant. Answer using the context below.
If the answer is not in the context, say: "I couldn't find that in the uploaded documents."
hi : I'M YOUR AI ASSISTANT.HOW CAN I HELP YOU?.
HELLO : I'M YOUR AI ASSISTANT.HOW CAN I HELP YOU?. 
Do NOT make up information. Search the relevant answer from the LLM model for the question.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=groq_api_key,
    )

    chain = (
        RunnableMap(
            {
                "context": RunnableLambda(
                    lambda q: "\n\n".join(
                        doc.page_content for doc in retriever.invoke(q)
                    )
                ),
                "question": RunnableLambda(lambda q: q),
            }
        )
        | prompt
        | llm
    )
    return chain, len(chunks)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Multi-PDF Chat Bot")
    st.caption("Upload PDFs and ask questions about their content.")
    st.divider()

    # API key input — pre-filled from .env, still editable if needed
    groq_key_input = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_…",
        value=st.session_state.groq_api_key,
        help="Loaded from .env automatically. Get your free key at console.groq.com",
    )
    if groq_key_input:
        st.session_state.groq_api_key = groq_key_input

    uploaded_pdfs = st.file_uploader(
        "📂 Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_btn = st.button("⚡ Process PDFs", use_container_width=True, type="primary")

    if process_btn:
        if not st.session_state.groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not uploaded_pdfs:
            st.error("❌ Please upload at least one PDF.")
        else:
            with st.status("Processing PDFs…", expanded=True) as status:
                st.write("📖 Reading PDF text…")
                raw_text = extract_text_from_pdfs(uploaded_pdfs)

                if not raw_text.strip():
                    status.update(label="❌ Failed", state="error")
                    st.error("No text found. Make sure your PDFs are not scanned images.")
                else:
                    st.write(f"✅ Extracted {len(raw_text):,} characters from {len(uploaded_pdfs)} file(s).")
                    st.write("🔄 Building vector store…")

                    try:
                        chain, num_chunks = build_rag_chain(
                            raw_text, st.session_state.groq_api_key
                        )
                        st.write(f"✅ Indexed {num_chunks} chunks into FAISS.")
                        st.session_state.rag_chain = chain
                        st.session_state.pdf_processed = True
                        st.session_state.messages = []
                        status.update(label="✅ Ready to chat!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label="❌ Build failed", state="error")
                        st.error(f"Error: {e}")

    st.divider()

    if st.session_state.pdf_processed:
        st.success("PDFs ready — start chatting! 🟢")
    else:
        st.info("Upload PDFs and click **Process PDFs** to begin.")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 Ask Your PDFs")

if not st.session_state.pdf_processed:
    st.info("👈 Upload your PDFs in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your documents…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = st.session_state.rag_chain.invoke(user_input)
                    answer = response.content
                except Exception as e:
                    answer = f"⚠️ Groq API error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})