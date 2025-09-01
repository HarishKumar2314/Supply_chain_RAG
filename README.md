# Supply Chain Risk Analysis Chatbot (RAG) — Streamlit

A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers supply-chain questions using your **Supplier** and **Tariff** documents. It returns the **top 5 matching excerpts** with sources, and (optionally) synthesizes an answer via OpenAI if you set `OPENAI_API_KEY` in `.env`.

## Quickstart

```bash
# 1) Create venv (recommended)
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate

# 2) Install
pip install -r requirements.txt

# 3) (Optional) Enable generation
cp .env.sample .env
# edit .env and set OPENAI_API_KEY

# 4) Run the app
streamlit run app/streamlit_app.py
```

## How to use
1. Place your documents under:
   - `data/suppliers/`
   - `data/tariffs/`
   (Supports `.txt`, `.md`, `.pdf`)
2. Click **(Re)build Index** in the sidebar (or upload additional files).
3. Ask a question. The app returns the **top 5** most relevant chunks with **sources & scores**.
4. If you provided an API key, you'll also get a **concise, cited answer** synthesized from those chunks.

## Notes
- Index is stored in `indices/` and auto-reused across sessions.
- No external vector DB required — uses **FAISS** locally.
- If you see errors reading a PDF, convert it to text first (or ensure it's not image-only).

## Project Structure
```
supplychain_rag_app/
├── app/
│   ├── streamlit_app.py
│   └── rag_utils.py
├── data/
│   ├── suppliers/
│   └── tariffs/
├── indices/
├── requirements.txt
└── .env.sample
```
