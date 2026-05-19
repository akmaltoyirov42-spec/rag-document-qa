"""RAG chain: retrieves context from the vector store, then asks the LLM."""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the
provided document context. If the answer is not in the context, say so clearly instead
of making something up. Keep answers concise and cite the page number when possible."""


def build_rag_chain(vector_store: FAISS, model: str = "gpt-4o-mini") -> ConversationalRetrievalChain:
    """
    Build a conversational RAG chain with memory.

    Uses a sliding window memory (last 5 turns) so the LLM doesn't lose
    context in long conversations without blowing up the token budget.
    """
    llm = ChatOpenAI(model=model, temperature=0)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",          # Max Marginal Relevance — reduces redundant chunks
        search_kwargs={"k": 4, "fetch_k": 10},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Send a question through the chain and return answer + source documents.

    Returns:
        {
            "answer": str,
            "sources": List[Document],
        }
    """
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result.get("source_documents", []),
    }
