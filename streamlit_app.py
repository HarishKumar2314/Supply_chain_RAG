import os
import pickle
import faiss
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

api_key = "AIzaSyAY3zR0AP9l7T7klLMQj2xEUsp_JZApi0E"

if not api_key:
    st.error("❌ GOOGLE_API_KEY is missing. Please provide your API key.")
else:
    genai.configure(api_key=api_key)
    st.success("✅ API key configured successfully!")

# === File paths for index persistence ===
INDEX_FILE = "indices/faiss.index"
META_FILE = "indices/metadata.pkl"

# === Utility: Save FAISS index + metadata ===
def build_index(uploaded_files):
    os.makedirs("indices", exist_ok=True)
    texts = []
    metadata = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            for i, row in df.iterrows():
                content = " ".join(map(str, row.values))
                texts.append(content)
                metadata.append({"file": uploaded_file.name, "row": i, "content": content})

        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            for i, line in enumerate(text.splitlines()):
                if line.strip():
                    texts.append(line)
                    metadata.append({"file": uploaded_file.name, "line": i, "content": line})

        elif uploaded_file.name.endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(uploaded_file)
            for i, page in enumerate(reader.pages):
                content = page.extract_text()
                if content:
                    texts.append(content)
                    metadata.append({"file": uploaded_file.name, "page": i, "content": content})

    # Build TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    # Build FAISS index
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump((metadata, vectorizer), f)

    return index, (metadata, vectorizer)

# === Load FAISS index ===
def load_index():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        obj = pickle.load(f)

        # Robust unpacking
        if isinstance(obj, tuple) and len(obj) == 2:
            metadata, vectorizer = obj
        else:
            metadata, vectorizer = obj, None

    return index, (metadata, vectorizer)

# === Retrieve top chunks ===
def retrieve_chunks(query, index, meta_tuple, top_k=5):
    metadata, vectorizer = meta_tuple
    if vectorizer is None:
        return [m["content"] for m in metadata[:top_k]]

    query_vec = vectorizer.transform([query]).toarray()
    D, I = index.search(query_vec, top_k)
    return [metadata[i]["content"] for i in I[0]]

# === Generate AI answer ===
def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a Supply Chain Risk Assistant.
Answer the question clearly with:
1. A **short professional summary paragraph**
2. A **well-formatted table** with supporting details from context.

Question: {query}

Context:
{context}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

st.set_page_config(page_title="Supply Chain Risk Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files (CSV, TXT, PDF)", 
        type=["csv", "txt", "pdf"], 
        accept_multiple_files=True
    )

    if st.button("⚡ Index uploaded files"):
        if uploaded_files:
            index, meta_tuple = build_index(uploaded_files)
            st.success("✅ Index built successfully!")
            st.session_state["index"] = index
            st.session_state["meta_tuple"] = meta_tuple
        else:
            st.warning("Please upload at least one file.")

    if st.button("🧹 Clear Index"):
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(META_FILE):
            os.remove(META_FILE)
        st.session_state.clear()
        st.info("Index cleared.")

# Load index if exists
if "index" not in st.session_state:
    index, meta_tuple = load_index()
    if index:
        st.session_state["index"] = index
        st.session_state["meta_tuple"] = meta_tuple

# Main app
st.title("📦 Supply Chain Risk Chatbot")
st.markdown("RAG with **FAISS + Local Embeddings + Gemini**")

query = st.text_input("💡 Ask a Question", placeholder="e.g. What are the biggest supplier risks?")
top_k = st.slider("🔎 Number of retrieved chunks", 1, 30, 5)

if st.button("🚀 Get Answer"):
    if "index" not in st.session_state or st.session_state["index"] is None:
        st.error("❌ No index found. Please upload and index files first.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            chunks = retrieve_chunks(query, st.session_state["index"], st.session_state["meta_tuple"], top_k)
            answer = generate_answer(query, chunks)

        st.subheader("📝 Answer")
        st.markdown(answer)
