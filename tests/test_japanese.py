"""Verify Japanese text transport with mock embeddings and real FAISS."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.ingest import load_documents
from src.llm import INSTRUCTIONS, OpenAIAnswerer, OpenAIEmbedder
from src.rag import RAGAssistant
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "水稲の出穂にはどのような気象条件が影響しますか？"


def test_japanese_retrieval_roundtrip_and_answer(tmp_path):
    """Check Unicode transport and ranking, not real embedding quality."""
    client = Mock()
    def embed(**kwargs):
        items = []
        for i, text in enumerate(kwargs['input']):
            vector = [1.0, 0.0] if '水稲の出穂と気温・日長' in text or text == QUESTION else [0.0, 1.0]
            items.append(SimpleNamespace(index=i, embedding=vector))
        return SimpleNamespace(data=items)
    client.embeddings.create.side_effect = embed
    client.responses.create.return_value.output_text = "デモ資料では気温と日長を扱っています [1]。"
    embedder = OpenAIEmbedder(client)
    retriever = Retriever.build(load_documents(ROOT / 'data/sample_documents'), embedder)
    retriever.save(tmp_path)
    restored = Retriever.load(tmp_path, embedder)
    result = RAGAssistant(restored, OpenAIAnswerer(client)).ask(QUESTION, 1)
    assert result.sources == ['rice_heading_ja.md']
    assert result.answer == 'デモ資料では気温と日長を扱っています [1]。'
    assert client.embeddings.create.call_args.kwargs['input'] == [QUESTION]
    prompt = client.responses.create.call_args.kwargs['input']
    assert QUESTION in prompt and '気温と日長' in prompt
    assert 'soil_monitoring.txt' not in prompt


def test_japanese_instructions_and_abstention():
    """Verify the required instruction and exact fallback through the pipeline."""
    assert 'Answer in Japanese by default' in INSTRUCTIONS
    assert 'answer in English' in INSTRUCTIONS
    assert '提供された資料からは確認できません' in INSTRUCTIONS
    retriever, answerer = Mock(), Mock()
    retriever.search.return_value = []
    result = RAGAssistant(retriever, answerer).ask(QUESTION)
    assert result.answer == '提供された資料からは確認できません'
    assert result.sources == []
    answerer.answer.assert_not_called()


def test_japanese_answer_and_source_ui(monkeypatch):
    """Exercise actual RAG composition and localized output in Streamlit."""
    from src.ingest import Chunk
    from src.retriever import SearchResult
    import src.llm
    import src.retriever

    client = Mock()
    client.responses.create.return_value.output_text = '気温と日長が挙げられています [1]。'
    retriever = Mock()
    retriever.search.return_value = [SearchResult(Chunk('気温と日長', 'rice_heading_ja.md', 0), 1.0)]
    monkeypatch.setattr(src.llm, 'create_client', lambda: client)
    monkeypatch.setattr(src.retriever.Retriever, 'load', lambda *args: retriever)
    app = AppTest.from_file(str(ROOT / 'app.py')).run()
    assert app.title[0].value == '農業技術資料 RAGアシスタント'
    question = next(x for x in app.text_input if x.label == '質問を入力してください')
    question.set_value(QUESTION)
    next(x for x in app.button if x.label == '質問する').click().run()
    assert not app.exception
    assert [x.value for x in app.subheader] == ['回答', '参照した資料']
    assert any('気温と日長' in x.value for x in app.markdown)
    assert any(x.value == 'rice_heading_ja.md' for x in app.text)
