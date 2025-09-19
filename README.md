"""
Financial Document Q&A Assistant
Single-file Streamlit app that:
 - Accepts PDF and Excel uploads
 - Extracts text and tables from documents
 - Builds a small retrieval context using TF-IDF
 - Sends the top context + user question to a local Ollama SLM for natural-language answers

How to use (summary):
 1. Install requirements:
    pip install -r requirements.txt
   (requirements.txt should include: streamlit, pdfplumber, pandas, numpy, scikit-learn, requests, openpyxl)

 2. Run Ollama locally and have a model available (example: "llama2" or another model you've installed).
    By default the app will call: http://localhost:11434/api/generate
    If your Ollama HTTP API uses a different URL, set environment variable OLLAMA_API_URL.
    Optionally set OLLAMA_MODEL to the model you want to call (default: "llama2")

 3. Start the app:
    streamlit run financial_qa_streamlit_app.py

Notes / limitations:
 - This is a demonstrator/prototype: it uses TF-IDF retrieval (fast, local) not embeddings.
 - PDF parsing with pdfplumber works well for text-based PDFs; scanned PDFs will require OCR (not included here).
 - Table extraction attempts best-effort; very complex, multi-column statements may require manual cleaning.
 - Ollama API payloads may vary depending on your installed Ollama version. Edit `ollama_generate` if your API differs.

"""

import os
import io
import re
import json
import traceback
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Utility: extraction functions
# -----------------------------

FINANCIAL_KEYWORDS = [
    "revenue", "sales", "net sales", "turnover", "income", "gross profit",
    "cost of goods", "cost of sales", "expenses", "operating expense", "ebitda",
    "profit", "net income", "income before tax", "tax", "assets", "liabilities",
    "equity", "cash flow", "cash", "receivables", "payables"
]


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file (best-effort)."""
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        traceback.print_exc()
    return "\n\n".join(text_parts)


def extract_tables_from_pdf_bytes(pdf_bytes: bytes) -> List[pd.DataFrame]:
    """Attempt to extract tables from PDF pages and return as DataFrames."""
    tables = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    page_tables = page.extract_tables()
                    for t in page_tables:
                        # convert to DataFrame, filter empty
                        df = pd.DataFrame(t)
                        # drop fully empty rows/cols
                        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
                        if df.shape[0] > 0 and df.shape[1] > 0:
                            tables.append(df)
                except Exception:
                    # continue if one page table extraction fails
                    continue
    except Exception as e:
        st.error(f"PDF table extraction error: {e}")
        traceback.print_exc()
    return tables


def extract_from_excel_bytes(excel_bytes: bytes) -> Tuple[str, List[pd.DataFrame]]:
    """Read Excel bytes into DataFrames for each sheet and produce a textual summary."""
    text_parts = []
    tables = []
    try:
        excel_file = io.BytesIO(excel_bytes)
        xls = pd.ExcelFile(excel_file)
        for sheet in xls.sheet_names:
            try:
                df = xls.parse(sheet_name=sheet, header=None)
                tables.append(df)
                text_parts.append(f"Sheet: {sheet}\n{df.head(20).to_csv(index=False, header=False)}")
            except Exception as e:
                text_parts.append(f"Could not parse sheet {sheet}: {e}")
    except Exception as e:
        st.error(f"Excel parsing error: {e}")
        traceback.print_exc()
    return "\n\n".join(text_parts), tables


# -----------------------------
# Utility: simple retrieval
# -----------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks (by characters)."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks


def build_tfidf_index(chunks: List[str]):
    """Return vectorizer and matrix for retrieval."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(chunks)
    return vectorizer, X


def retrieve_top_chunks(question: str, chunks: List[str], vectorizer, X, top_k: int = 3) -> List[Tuple[int, float, str]]:
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, X).flatten()
    idx = np.argsort(sims)[::-1][:top_k]
    results = [(int(i), float(sims[i]), chunks[i]) for i in idx if sims[i] > 0]
    return results


# -----------------------------
# Ollama communication helper
# -----------------------------

def ollama_generate(prompt: str, model: str = None, api_url: str = None, max_tokens: int = 512) -> str:
    """
    Sends prompt to a local Ollama HTTP API using a generic /api/generate interface.
    If your Ollama HTTP API has a different route, update this function accordingly.
    """
    if api_url is None:
        api_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
    if model is None:
        model = os.environ.get("OLLAMA_MODEL", "llama2")

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Try to be flexible with response shape
        if isinstance(data, dict):
            # common: {"choices": [{"text": "..."}]}
            if "choices" in data and len(data["choices"]) > 0:
                txt = data["choices"][0].get("text") or data["choices"][0].get("content")
                if txt:
                    return txt
            # attempt other keys
            if "text" in data:
                return data["text"]
            return json.dumps(data)
        else:
            return str(data)
    except Exception as e:
        st.error(f"Error contacting Ollama API: {e}")
        traceback.print_exc()
        return ""


# -----------------------------
# Financial helpers (very small heuristics)
# -----------------------------

def find_key_lines(text: str, keywords=FINANCIAL_KEYWORDS, context_lines: int = 2) -> str:
    """Return lines around keywords to present as quick summary."""
    lines = text.splitlines()
    selected = []
    for i, line in enumerate(lines):
        low = line.lower()
        for kw in keywords:
            if kw in low:
                start = max(i - context_lines, 0)
                end = min(i + context_lines + 1, len(lines))
                snippet = "\n".join(lines[start:end])
                selected.append(snippet)
                break
    return "\n\n".join(selected)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Financial Document Q&A", layout="wide")
st.title("📄 Financial Document Q&A Assistant")
st.markdown(
    "Upload PDF or Excel financial statements, then ask natural-language questions about revenue, expenses, profit, etc. This is a local prototype that uses Ollama for LLM reasoning and a TF-IDF retriever for document context."
)

# sidebar
st.sidebar.header("Settings")
ollama_url = st.sidebar.text_input("Ollama API URL", value=os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate"))
ollama_model = st.sidebar.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "llama2"))
max_tokens = st.sidebar.slider("Max tokens for LLM response", min_value=128, max_value=2048, value=512)

if "docs" not in st.session_state:
    st.session_state.docs = []  # list of dicts: {name, text, tables, chunks, vectorizer, X}

uploaded = st.file_uploader("Upload financial document (PDF or Excel)", type=["pdf", "xls", "xlsx"], accept_multiple_files=True)

if uploaded:
    for up in uploaded:
        if any(d.get("name") == up.name for d in st.session_state.docs):
            st.info(f"Already uploaded: {up.name}")
            continue
        st.info(f"Processing {up.name}...")
        try:
            raw = up.read()
            if up.name.lower().endswith('.pdf'):
                text = extract_text_from_pdf_bytes(raw)
                tables = extract_tables_from_pdf_bytes(raw)
            else:
                text, tables = extract_from_excel_bytes(raw)

            if not text or text.strip() == "":
                st.warning(f"No text extracted from {up.name}. If this is a scanned document, OCR is required (not included).")

            chunks = chunk_text(text)
            if len(chunks) == 0:
                chunks = [text]
            try:
                vectorizer, X = build_tfidf_index(chunks)
            except Exception:
                vectorizer, X = None, None

            st.session_state.docs.append({
                "name": up.name,
                "text": text,
                "tables": tables,
                "chunks": chunks,
                "vectorizer": vectorizer,
                "X": X
            })
            st.success(f"Loaded {up.name}")
        except Exception as e:
            st.error(f"Failed to process {up.name}: {e}")

# show loaded docs
if len(st.session_state.docs) == 0:
    st.info("No documents loaded yet. Upload a PDF or Excel file from the panel above.")
else:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Loaded documents")
        for i, d in enumerate(st.session_state.docs):
            if st.button(f"Show summary: {d['name']}", key=f"show_{i}"):
                st.session_state.show_doc = i
        if st.button("Clear documents"):
            st.session_state.docs = []
            st.experimental_rerun()
    with col2:
        idx = st.session_state.get("show_doc", 0)
        if idx < len(st.session_state.docs):
            doc = st.session_state.docs[idx]
            st.subheader(f"Preview — {doc['name']}")
            st.markdown("**Key lines (heuristic search):**")
            st.code(find_key_lines(doc['text'])[:4000])
            if len(doc['tables']) > 0:
                st.markdown("**Detected tables (first few rows):**")
                for j, t in enumerate(doc['tables'][:3]):
                    st.text(f"Table {j+1} shape: {t.shape}")
                    try:
                        # show small preview
                        st.dataframe(t.head(10))
                    except Exception:
                        st.text(t.to_string())

# Chat interface
st.markdown("---")
st.subheader("Ask questions")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)

question = st.text_input("Enter your question about the uploaded documents")
col_a, col_b = st.columns([1, 3])
with col_a:
    selected_doc = st.selectbox("Use document", options=[d['name'] for d in st.session_state.docs] if st.session_state.docs else [])
    top_k = st.number_input("Context chunks to retrieve", min_value=1, max_value=10, value=3)
    show_context = st.checkbox("Show retrieved context", value=True)
with col_b:
    submit = st.button("Ask")

if submit and question:
    if not st.session_state.docs:
        st.error("No documents uploaded — please upload at least one PDF or Excel file.")
    else:
        # find selected doc
        doc = next((d for d in st.session_state.docs if d['name'] == selected_doc), st.session_state.docs[0])
        # retrieval
        retrieved_text = ""
        if doc['vectorizer'] is not None and doc['X'] is not None:
            results = retrieve_top_chunks(question, doc['chunks'], doc['vectorizer'], doc['X'], top_k=top_k)
            if results:
                retrieved_text = "\n\n".join([f"[score={s:.3f}] {c}" for (_i, s, c) in results for c in [c if isinstance(c, str) else str(c)])
        else:
            # fallback: simple heuristic
            retrieved_text = find_key_lines(doc['text'])

        # build prompt
        prompt_parts = []
        prompt_parts.append("You are a helpful financial analysis assistant. Answer concisely and cite the document when possible.")
        prompt_parts.append("Document name: " + doc['name'])
        prompt_parts.append("Extracted snippets:")
        prompt_parts.append(retrieved_text[:8000])
        prompt_parts.append("User question:")
        prompt_parts.append(question)
        prompt = "\n\n".join(prompt_parts)

        if show_context:
            st.markdown("**Context sent to the LLM (truncated):**")
            st.code(retrieved_text[:3000])

        st.markdown("**Answer:**")
        answer = ollama_generate(prompt=prompt, model=ollama_model, api_url=ollama_url, max_tokens=max_tokens)
        if answer:
            st.write(answer)
            st.session_state.chat_history.append((question, answer))
        else:
            st.write("(No answer received from local LLM.)")

# Show conversation
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Conversation history")
    for q, a in reversed(st.session_state.chat_history[-10:]):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")

# Footer / quick tips
st.markdown("---")
st.markdown("**Quick tips:** PDFs that are scanned require OCR (Tesseract) before text extraction. For best results, upload statement PDFs that are text-based or well-structured Excel exports of statements.")


# If run as script - nothing extra
if __name__ == '__main__':
    pass
