"""
RAG Bot with a web UI (using Streamlit).
Answers questions using YOUR OWN documents, shown in a browser instead of the terminal.

Setup:
1. pip install streamlit google-genai chromadb sentence-transformers pypdf
2. Get a free API key at https://aistudio.google.com -> "Get API key"
3. Set your API key:
   PowerShell: $env:GOOGLE_API_KEY="your-key-here"
4. Run: streamlit run rag_bot_ui.py
   (Note: NOT "python rag_bot_ui.py" -- Streamlit apps are launched differently)
5. A browser tab will open automatically at http://localhost:8501

Note: Google deprecated the old "google-generativeai" package in 2026.
This script uses the new, current "google-genai" package instead.

How this differs from the terminal version:
- Documents are uploaded through the browser instead of read from a folder
- st.session_state keeps data (like the vector database and chat history)
  around between interactions, since Streamlit reruns the whole script on
  every click/input -- this is the biggest mental shift from a normal script
"""

import os
import streamlit as st
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

CHUNK_SIZE = 500


def extract_text_from_pdf(uploaded_file):
    """Read all pages of an uploaded PDF and return their combined text."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---- Page setup ----
st.set_page_config(page_title="RAG Bot", page_icon="📚")
st.title("📚 Chat With Your Documents")

# ---- API key handling ----
api_key = os.environ.get("GOOGLE_API_KEY", "")
if not api_key:
    api_key = st.text_input("Enter your Google API key", type="password")
    if not api_key:
        st.info("Get a free key at https://aistudio.google.com, or set GOOGLE_API_KEY in your terminal before launching.")
        st.stop()

genai_client = genai.Client(api_key=api_key)


# ---- Cache expensive setup so it doesn't reload on every interaction ----
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.Client()
    return chroma_client.get_or_create_collection(name="my_documents")


embedder = load_embedder()
collection = get_chroma_collection()
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# ---- Session state: tracks whether documents have been indexed yet ----
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---- Sidebar: upload and index documents ----
with st.sidebar:
    st.header("1. Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Index documents", disabled=not uploaded_files):
        chunks = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            for i in range(0, len(text), CHUNK_SIZE):
                chunk = text[i:i + CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(chunk)

        if chunks:
            embeddings = embedder.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            collection.add(documents=chunks, embeddings=embeddings, ids=ids)
            st.session_state.indexed = True
            st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed:
        st.success("✅ Documents ready — ask questions on the right.")
    else:
        st.warning("⬆️ Upload and index documents to get started.")


# ---- Main area: chat interface ----
st.header("2. Ask questions")

# Display past messages
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)

# New question input
question = st.chat_input(
    "Ask something about your documents...",
    disabled=not st.session_state.indexed
)

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.write(question)

    # Retrieve relevant chunks
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)
    relevant_chunks = results["documents"][0]

    # Build prompt and call Gemini
    context_text = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say you don't know based on the provided documents.

Context:
{context_text}

Question: {question}"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt
            )
            answer = response.text
            st.write(answer)

        with st.expander("Show retrieved context"):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.markdown(f"**Chunk {i}:** {chunk}")

    st.session_state.chat_history.append(("assistant", answer))