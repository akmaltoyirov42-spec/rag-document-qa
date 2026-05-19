# RAG Document Q&A

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)

Upload a PDF, ask questions about it, get answers with page citations. Uses RAG so the LLM only answers from what's actually in your document.

---

## How it works

PDF gets split into chunks → each chunk is embedded with OpenAI → stored in FAISS. When you ask a question, the closest chunks get retrieved and sent to GPT-4o-mini as context.

The chain keeps the last 5 messages in memory so follow-up questions work properly.

---

## Setup

```bash
git clone https://github.com/akmaltoyirov42-spec/rag-document-qa.git
cd rag-document-qa

python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your OpenAI API key to .env

streamlit run app.py
```

You need an OpenAI API key. Processing a 50-page PDF and asking ~20 questions costs around $0.01–0.03.

## Docker

```bash
docker build -t docchat .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... docchat
```

## Tests

```bash
pytest tests/ -v
```

---

## Stack

Python, LangChain, FAISS, OpenAI embeddings, GPT-4o-mini, Streamlit, Docker
