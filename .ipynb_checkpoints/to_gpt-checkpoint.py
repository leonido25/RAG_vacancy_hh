# --- простой токенайзер (для русского норм работает в Py3) ---
def tokenize(text):
    return re.findall(r"\w+", text.lower())

# --- пример класса-ретривера ---
class BM25Retriever(Runnable):
    def __init__(self, docs, top_k=6):
        """
        docs: список langchain Document (или объектов с page_content + metadata)
        top_k: сколько возвращать
        """
        super().__init__()
        self.docs = docs
        self.top_k = top_k
        self.corpus = [d.page_content for d in docs]
        self.tokenized_corpus = [tokenize(t) for t in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_top_docs_with_scores(self, query, topn=None):
        topn = topn or self.top_k
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)            # numpy array
        # используем argpartition для скорости, потом сортируем выбранные
        if len(scores) <= topn:
            top_idx = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(-scores, topn)[:topn]
            top_idx = part[np.argsort(scores[part])[::-1]]
        result = []
        for i in top_idx:
            doc = self.docs[i]
            score = float(scores[i])
            # записываем score в metadata (полезно дальше)
            try:
                doc.metadata["bm25_score"] = score
            except Exception:
                # если metadata нет или immutable, создаём новый Document
                doc = Document(page_content=doc.page_content,
                               metadata={**getattr(doc, "metadata", {}), "bm25_score": score})
            result.append((doc, score))
        return result

    def get_relevant_documents(self, query):
        return [d for d, s in self.get_top_docs_with_scores(query)]

    # Совместимость с Runnable API (проверить сигнатуру в твоей версии)
    def invoke(self, query):
        return self.get_relevant_documents(query)

    def __call__(self, query):
        return self.invoke(query)


# --- форматирование контекста, включающее источники ---
def format_docs(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("id") or d.metadata.get("source") or d.metadata.get("vacancy_id") or "unknown_id"
        parts.append(f"[Источник: {src}]\n{d.page_content}")
    return "\n\n".join(parts)

simple_docs = df_to_langchain_documents(df_new)
retriever = BM25Retriever(simple_docs, top_k=6)
docs = retriever.get_relevant_documents(query_test)
context = format_docs(docs)
# LLM
llm = OllamaLLM(model="llama3", temperature=0)

#### RETRIEVAL and GENERATION ####

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
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)


query_test = "Какие есть вакансии Data Scientist санкт петербурге? "

final_prompt = prompt.format_prompt(context=context, question=query_test).to_messages()
response = llm.generate(final_prompt)  # зависит от API LLM
# Chain
rag_chain_smart = (
    {
        # Шаг 1: Поиск контекста (вопрос -> retriever -> форматирование)
        "context": itemgetter("question") | retriever | format_docs,  
        # Шаг 2: Сохранение исходного вопроса
        "question": itemgetter("question")
    }
    | prompt          # Шаг 3: Применение промпта к контексту и вопросу
    | llm             # Шаг 4: Передача в модель Gemini
    | StrOutputParser() # Шаг 5: Получение ответа в виде строки
)
rag_chain_smart.invoke({"question": query_test})