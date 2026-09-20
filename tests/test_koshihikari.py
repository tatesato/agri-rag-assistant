"""Check the demo's evidence flow with mock APIs and real FAISS."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

from streamlit.testing.v1 import AppTest

from src.ingest import load_documents
from src.llm import INSTRUCTIONS, OpenAIAnswerer, OpenAIEmbedder
from src.rag import RAGAssistant
from src.retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTION = 'コシヒカリの出穂予測では、気温と日長をどのように利用しますか？'


def test_koshihikari_evidence_roundtrip(tmp_path):
    client = Mock()
    client.embeddings.create.side_effect = lambda **kw: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[1., 0.] if 'コシヒカリ' in t else [0., 1.])
        for i, t in enumerate(kw['input'])])
    client.responses.create.return_value.output_text = 'デモ設計では気温と日長からDVRを求め、DVIに積算します[1]。'
    chunks = load_documents(ROOT / 'data/sample_documents')
    embedder = OpenAIEmbedder(client)
    retriever = Retriever.build(chunks, embedder)
    retriever.save(tmp_path)
    response = RAGAssistant(Retriever.load(tmp_path, embedder), OpenAIAnswerer(client)).ask(QUESTION, 1)
    assert response.sources == ['koshihikari_heading_ja.md']
    prompt = client.responses.create.call_args.kwargs['input']
    excerpts = json.loads(prompt.split('Retrieved excerpts:\n', 1)[1].split('\n\nQuestion:', 1)[0])
    assert excerpts[0]['text'] == response.results[0].chunk.text
    assert 'ベイズ最適化' in excerpts[0]['text']
    assert '実測出穂日と予測出穂日を比較' in excerpts[0]['text']
    assert '実証済み' in excerpts[0]['text']
    assert 'Do not use outside knowledge, speculate, or invent facts.' in INSTRUCTIONS
    assert 'a demo design is not a validated research result' in INSTRUCTIONS


def test_unsupported_parameter_answer_preserves_fallback():
    """Verify fallback transport; a mock does not prove live model grounding."""
    client = Mock()
    client.responses.create.return_value.output_text = '提供された資料からは確認できません'
    retriever = Mock()
    from src.retriever import SearchResult
    chunk = next(c for c in load_documents(ROOT / 'data/sample_documents')
                 if c.source == 'koshihikari_heading_ja.md')
    retriever.search.return_value = [SearchResult(chunk, 1.)]
    result = RAGAssistant(retriever, OpenAIAnswerer(client)).ask('コシヒカリの最適パラメータの数値は？')
    assert result.answer == '提供された資料からは確認できません'
    assert result.sources == []


def test_koshihikari_demo_question_ui():
    app = AppTest.from_file(str(ROOT / 'app.py')).run()
    assert not app.exception
    question = next(t for t in app.text_input if t.label == '質問を入力してください')
    assert question.placeholder == QUESTION
    assert any(QUESTION in c.value for c in app.caption)
