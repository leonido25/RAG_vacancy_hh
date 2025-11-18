# app.py
import streamlit as st
import rag_pipeline  
from typing import Any, Dict

st.set_page_config(page_title="Simple RAG", layout="centered")
st.title("RAG — Вопрос → Ответ")

st.markdown("Введите вопрос в поле ниже и нажмите **Задать вопрос**")

# Боковая панель — настройки
with st.sidebar:
    st.header("Настройки")
    show_sources = st.checkbox("Показывать найденные документы (источники)", value=True)
    max_chars = st.slider("Макс. символов для предпросмотра источника", 100, 5000, 800)
    st.write("RAG_USE_MOCK (в окружении) управляет использованием мок-режима.")

# Поле ввода
query = st.text_area("Вопрос", height=140, placeholder="Например: 'Что такое RAG?'")

col1, col2 = st.columns([1, 3])
with col1:
    submit = st.button("Задать вопрос")
with col2:
    clear = st.button("Очистить поле")

if clear:
    st.experimental_set_query_params()  # просто очистит параметры URL, полезно на будущее
    st.experimental_rerun()

def parse_response(resp: Any) -> Dict[str, Any]:
    """
    Универсальный парсер: пытается извлечь текст ответа и source_documents
    Поддерживает dict, объекты с атрибутами или строки.
    """
    out = {"answer": None, "sources": None, "raw": resp}
    if resp is None:
        return out
    if isinstance(resp, dict):
        out["answer"] = resp.get("answer") or resp.get("output_text") or resp.get("text") or str(resp)
        out["sources"] = resp.get("source_documents") or resp.get("sources") or resp.get("documents")
    else:
        # Попробуем у объекта атрибут answer и source_documents
        ans = getattr(resp, "answer", None) or getattr(resp, "text", None)
        src = getattr(resp, "source_documents", None) or getattr(resp, "sources", None)
        out["answer"] = ans or str(resp)
        out["sources"] = src
    return out

if submit:
    if not query.strip():
        st.warning("Пожалуйста, введите вопрос.")
    else:
        # Кэширование самой загрузки пайплайна решено сделать внутри rag_pipeline (мы импортировали модуль)
        with st.spinner("Выполняется поиск контекста и генерация ответа..."):
            try:
                raw_resp = rag_pipeline.answer_question(query)
            except Exception as e:
                st.error(f"Ошибка при вызове пайплайна: {e}")
                st.stop()

        parsed = parse_response(raw_resp)
        st.markdown("### Ответ")
        st.write(parsed["answer"] or "(пустой ответ)")

        if show_sources:
            sources = parsed.get("sources")
            if sources:
                st.markdown("### Найденные документы")
                # ожидаем список документов (dict-ы или объекты)
                for i, doc in enumerate(sources):
                    if isinstance(doc, dict):
                        content = doc.get("page_content") or doc.get("content") or ""
                        metadata = doc.get("metadata", {})
                    else:
                        content = getattr(doc, "page_content", None) or str(doc)
                        metadata = getattr(doc, "metadata", {}) or {}
                    snippet = (content[:max_chars] + ("..." if len(content) > max_chars else "")) if content else "(нет контента)"
                    with st.expander(f"Документ {i+1} — {metadata.get('source','unknown')}"):
                        st.write(snippet)
                        if metadata:
                            st.write("**Метаданные:**")
                            st.json(metadata)
            else:
                st.info("Источники не найдены или не возвращены пайплайном.")

        # Покажем сырые данные для отладки
        with st.expander("Показать полные сырые данные ответа (debug)"):
            st.json(parsed["raw"])
