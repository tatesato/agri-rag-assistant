"""Streamlit entry point for agricultural document question answering."""
import os
from pathlib import Path

import streamlit as st

from src.ingest import load_documents
from src.diagnostics import safe_traceback
from src.llm import OpenAIAnswerer, OpenAIEmbedder, create_client
from src.rag import RAGAssistant
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "index"


def main() -> None:
    """Build or load the index and display grounded answers with evidence."""
    st.set_page_config(page_title="農業技術資料 RAGアシスタント", page_icon="🌱")
    st.title("農業技術資料 RAGアシスタント")
    st.caption("コシヒカリの出穂予測を題材に、農業技術資料を検索し、根拠を参照して回答します。")
    st.info("サンプル資料はデモ用です。索引作成と質問時には文章をOpenAIへ送信します。")
    directory = st.sidebar.text_input("資料フォルダ", str(ROOT / "data/sample_documents"))
    top_k = st.sidebar.slider("検索する資料断片の数（Top-K）", 1, 10, 3)
    build = st.sidebar.button("索引を作成・更新")
    question = st.text_input("質問を入力してください", placeholder="コシヒカリの出穂予測では、気温と日長をどのように利用しますか？")
    st.caption("質問例：コシヒカリの出穂予測では、気温と日長をどのように利用しますか？")
    submit = st.button("質問する")
    if not (build or submit):
        return
    try:
        if submit and not question.strip():
            st.warning("質問を入力してください。")
            return
        client = create_client()
        embedder = OpenAIEmbedder(client, os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
        if build:
            with st.spinner("資料を読み込み、索引を作成しています…"):
                chunks = load_documents(Path(directory))
                retriever = Retriever.build(chunks, embedder)
                retriever.save(INDEX)
            st.success(f"{len(chunks)}件の資料断片を登録しました。資料や埋め込みモデルを変更した場合は索引を更新してください。")
        if submit:
            with st.spinner("関連資料を検索し、回答を作成しています…"):
                retriever = Retriever.load(INDEX, embedder)
                answerer = OpenAIAnswerer(client, os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
                response = RAGAssistant(retriever, answerer).ask(question, top_k)
            st.subheader("回答")
            st.markdown(response.answer)
            st.subheader("参照した資料")
            if response.sources:
                for source in response.sources:
                    st.text(source)
            else:
                st.caption("回答に引用がありません。以下の検索結果を確認してください。")
            for i, result in enumerate(response.results, 1):
                with st.expander(f"[{i}] {result.chunk.source} · 類似度 {result.score:.3f}"):
                    st.text(result.chunk.text)
    except Exception as exc:
        import traceback
        traceback.print_exc()

        st.error(
            "処理できませんでした。APIキー、モデルの利用権限、資料フォルダ、"
            "通信状態を確認してください。質問する前に索引を作成・更新してください。"
        )

        if os.getenv("APP_ENV") == "development":
            with st.expander("開発用エラー詳細", expanded=True):
                st.code(safe_traceback(exc), language="text")

if __name__ == "__main__":
    main()
