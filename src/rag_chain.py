from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


def build_rag_chain(vector_store: FAISS, model: str = "gpt-4o-mini") -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model=model, temperature=0)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,
    )

    # MMR reduces redundant chunks when multiple pages say the same thing
    retriever = vector_store.as_retriever(
        search_type="mmr",
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
    result = chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result.get("source_documents", []),
    }
