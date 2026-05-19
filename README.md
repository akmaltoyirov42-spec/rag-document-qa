# 📄 DocChat — RAG-Powered PDF Q&A System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-orange)
![OpenAI](https://img.shields.io/badge/LLM-GPT--4o--mini-412991?logo=openai)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Upload any PDF and have a conversation with it. Ask questions, get answers grounded in the document, with source citations.

---

## Demo

<!-- Add a GIF here after recording your screen: https://www.screentogif.com/ -->
![Demo](assets/demo.gif)

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐     embed      ┌──────────────┐
│  Streamlit  │ ──────────────▶│   OpenAI     │
│  Front-end  │                │  Embeddings  │
└─────────────┘                └──────┬───────┘
                                      │ vector
                                      ▼
                               ┌──────────────┐   top-4 chunks
                               │ FAISS Index  │ ──────────────────┐
                               └──────────────┘                   │
                                                                   ▼
                                                          ┌────────────────┐
                                                          │  GPT-4o-mini   │
                                                          │  (LangChain)   │
                                                          └────────┬───────┘
                                                                   │
                                                                   ▼
                                                              Answer + Sources
```

**Flow:**
1. **Ingest** — PDF is parsed page-by-page with `pypdf`
2. **Chunk** — Text is split into 1 000-character chunks with 200-char overlap (`RecursiveCharacterTextSplitter`)
3. **Embed** — Each chunk is embedded with `text-embedding-3-small` (OpenAI)
4. **Index** — Embeddings are stored in a FAISS in-memory vector store
5. **Retrieve** — User question is embedded → top-4 relevant chunks fetched via MMR (Max Marginal Relevance)
6. **Generate** — GPT-4o-mini generates a grounded answer with page citations
7. **Memory** — Conversation history (last 5 turns) is kept for follow-up questions

---

## Features

- **Multi-turn conversations** — ask follow-up questions; the model remembers context
- **Source citations** — every answer shows which page the information came from
- **MMR retrieval** — avoids returning redundant chunks; improves answer quality
- **No data stored** — PDFs are processed in-memory and never saved to disk
- **Docker-ready** — one command to run anywhere
- **Extensible** — swap GPT-4o-mini for any other LangChain-compatible LLM

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector DB | FAISS (Meta) |
| Orchestration | LangChain 0.3 |
| PDF Parsing | pypdf |
| Front-end | Streamlit |
| Containerization | Docker |

---

## Getting Started

### Prerequisites
- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
```

### 2. Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Alternative: Run with Docker
```bash
docker build -t docchat .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... docchat
```

---

## Running Tests
```bash
pip install pytest
pytest tests/ -v
```

---

## Project Structure
```
rag-document-qa/
├── src/
│   ├── document_processor.py  # PDF loading & text chunking
│   ├── vector_store.py        # FAISS index management
│   └── rag_chain.py           # LangChain RAG pipeline
├── tests/
│   └── test_document_processor.py
├── app.py                     # Streamlit UI
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Cost Estimate

Using GPT-4o-mini and text-embedding-3-small, processing a 50-page PDF and asking ~20 questions costs approximately **$0.01–0.03**. Very affordable for development and demos.

---

## What I Learned

- How **RAG** solves the hallucination problem by grounding LLM answers in retrieved facts
- How **vector similarity search** works with FAISS (HNSW index, cosine similarity)
- Why **chunk overlap** matters — without it, key sentences at chunk boundaries are missed
- How **MMR** (Max Marginal Relevance) improves retrieval by balancing relevance vs. diversity
- LangChain's `ConversationalRetrievalChain` and conversation memory management

---

## License

MIT © Akmal Toyirov
