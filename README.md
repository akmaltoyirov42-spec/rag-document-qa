# RAG document Q&A

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red)

upload a PDF, ask questions about it, get answers with page citations. uses RAG so the LLM only answers from what's actually in the document instead of making stuff up.

---

## how it works

PDF gets split into chunks → each chunk gets embedded with OpenAI → stored in a FAISS index. when you ask a question, the closest chunks get pulled out and stuffed into a GPT-4o-mini prompt as context.

the chain remembers the last 5 messages so follow-up questions like "what about the second point" actually work.

---

## setup

```bash
git clone https://github.com/akmaltoyirov42-spec/rag-document-qa.git
cd rag-document-qa

python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your OpenAI key to .env

streamlit run app.py
```

needs an OpenAI API key. processing a 50-page PDF and asking ~20 questions costs about $0.01–0.03.

## docker

```bash
docker build -t docchat .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... docchat
```

## tests

```bash
pytest tests/ -v
```

---

## notes

- chunk size matters a lot — too small loses context, too big retrieves irrelevant stuff
- MMR retrieval (instead of plain similarity) gave better answers because it avoids duplicate chunks
- without conversational memory the bot can't handle follow-ups properly
- citing the page number makes the output way more trustworthy than just text

---

## what's next

want to swap OpenAI embeddings for a local model (BGE or e5) so the whole thing can run offline. also planning to add hybrid search (BM25 + vector) which usually beats pure vector search on technical documents.

---

Python, LangChain, FAISS, OpenAI embeddings, GPT-4o-mini, Streamlit, Docker
