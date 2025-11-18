
from operator import itemgetter
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import logging

# 1) эмбеддинги 
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 2) загрузка FAISS 
loaded_vectorstore_simple = FAISS.load_local(
    "faiss_index_final",  
    embeddings,
    allow_dangerous_deserialization=True
)

# 3) ретривер
retriever_simple = loaded_vectorstore_simple.as_retriever(search_kwargs={"k": 8})

# 4) форматирование найденных документов 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

system_template = (
"Ты — HR-помощник, специалист по подбору персонала.\n\n"
    "Инструкции:\n"
    "1. Отвечай ТОЛЬКО на основе предоставленного контекста\n"
    "2. Структурируй ответ на русском языке\n"
    "3. ВСЕГДА указывай источники в формате [Источник: ID_вакансии]\n"
    "4. Если информации недостаточно - прямо скажи об этом\n\n"
    "Контекст: {context}\n\n"
    "Вопрос: {question}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{question}"),
    ]
)


llm = OllamaLLM(model="llama3", temperature=0)

from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {
        "context": itemgetter("question") | retriever_simple | format_docs,
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Обёртка для UI
def answer_question(question: str):
    """
    Вызов пайплайна. Возвращает dict с ключами:
    """
    try:
        result = rag_chain.invoke({"question": question})
    except Exception as e:
        logging.exception("Ошибка вызова rag_chain")
        return {"answer": None, "source_documents": [], "raw": {"error": str(e)}}

    # попытка нормализовать ответ в dict
    if isinstance(result, dict):
        out = result
    else:
        # некоторые раннеры возвращают объект с полем text/answer
        out = {}
        out["answer"] = getattr(result, "text", None) or getattr(result, "answer", None) or str(result)
        out["source_documents"] = getattr(result, "source_documents", None) or getattr(result, "sources", None) or []
        out["raw"] = result

    return out
