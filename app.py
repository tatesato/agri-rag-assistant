"""Streamlit entry point for agricultural document question answering."""
import os
from pathlib import Path

import streamlit as st

from src.ingest import load_documents
from src.llm import OpenAIAnswerer, OpenAIEmbedder, create_client
from src.rag import RAGAssistant
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "index"


def main() -> None:
    """Build or load the index and display grounded answers with evidence."""
    st.set_page_config(page_title="Agri RAG Assistant", page_icon="🌱")
    st.title("🌱 Agri RAG Assistant")
    st.caption("Ask agricultural questions and inspect the supporting documents.")
    st.info("Demo documents are illustrative. Building and querying send text to OpenAI.")
    directory = st.sidebar.text_input("Document directory", str(ROOT / "data/sample_documents"))
    top_k = st.sidebar.slider("Top-K excerpts", 1, 10, 3)
    build = st.sidebar.button("Build / rebuild index")
    question = st.text_input("Question", placeholder="What should I record before irrigation?")
    submit = st.button("Ask")
    if not (build or submit):
        return
    try:
        if submit and not question.strip():
            st.warning("Enter a question first.")
            return
        client = create_client()
        embedder = OpenAIEmbedder(client, os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        if build:
            with st.spinner("Reading documents and building the index..."):
                chunks = load_documents(Path(directory))
                retriever = Retriever.build(chunks, embedder)
                retriever.save(INDEX)
            st.success(f"Indexed {len(chunks)} chunks. Rebuild after changing documents or embedding model.")
        if submit:
            with st.spinner("Retrieving evidence and generating an answer..."):
                retriever = Retriever.load(INDEX, embedder)
                answerer = OpenAIAnswerer(client, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
                response = RAGAssistant(retriever, answerer).ask(question, top_k)
            st.markdown(response.answer)
            st.subheader("Cited source files")
            if response.sources:
                for source in response.sources:
                    st.text(source)
            else:
                st.caption("No explicit citations were returned. Review the evidence below.")
            for i, result in enumerate(response.results, 1):
                with st.expander(f"[{i}] {result.chunk.source} · similarity {result.score:.3f}"):
                    st.text(result.chunk.text)
    except (ValueError, RuntimeError, OSError):
        st.error("Unable to complete the request. Check your .env API key, model access, "
                 "UTF-8 document directory, and network connection. Build or rebuild the index before asking.")


if __name__ == "__main__":
    main()
