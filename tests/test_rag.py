"""Test evidence isolation, citations and OpenAI adapters without API calls."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from openai import OpenAIError
import pytest

from src.ingest import Chunk
from src.llm import OpenAIAnswerer, OpenAIEmbedder, create_client
from src.rag import RAGAssistant
from src.retriever import SearchResult


def test_context_contains_only_search_results_and_sources_are_deduplicated():
    hits = [SearchResult(Chunk("moisture log", "water.md", 0), 0.9),
            SearchResult(Chunk("time log", "water.md", 1), 0.8)]
    searcher = Mock()
    searcher.search.return_value = hits
    answerer = Mock()
    answerer.answer.return_value = "Record moisture [1] and time [2]."
    response = RAGAssistant(searcher, answerer).ask("What to record?", 2)
    searcher.search.assert_called_once_with("What to record?", 2)
    question, context = answerer.answer.call_args.args
    assert question == "What to record?"
    assert [item["text"] for item in json.loads(context)] == ["moisture log", "time log"]
    assert response.sources == ["water.md"]
    assert response.results == hits


def test_no_results_skips_llm():
    searcher, answerer = Mock(), Mock()
    searcher.search.return_value = []
    result = RAGAssistant(searcher, answerer).ask("question")
    assert result.sources == []
    answerer.answer.assert_not_called()


@pytest.mark.parametrize("answer", ["Claim [0]", "Claim [2]"])
def test_unknown_citation_is_rejected(answer):
    searcher, answerer = Mock(), Mock()
    searcher.search.return_value = [SearchResult(Chunk("text", "a.md", 0), 1)]
    answerer.answer.return_value = answer
    with pytest.raises(ValueError, match="citation"):
        RAGAssistant(searcher, answerer).ask("question")


def test_abstention_does_not_invent_sources():
    searcher, answerer = Mock(), Mock()
    searcher.search.return_value = [SearchResult(Chunk("text", "a.md", 0), 1)]
    answerer.answer.return_value = "The supplied evidence is insufficient."
    assert RAGAssistant(searcher, answerer).ask("question").sources == []


def test_blank_question_skips_dependencies():
    searcher, answerer = Mock(), Mock()
    with pytest.raises(ValueError):
        RAGAssistant(searcher, answerer).ask(" ")
    searcher.search.assert_not_called()
    answerer.answer.assert_not_called()


def test_embedding_batching_and_order():
    client = Mock()
    def respond(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i), 1.0])
                                     for i in reversed(range(len(kwargs["input"])))])
    client.embeddings.create.side_effect = respond
    vectors = OpenAIEmbedder(client).embed(["text"] * 65)
    assert len(vectors) == 65
    assert vectors[0] == [0, 1]
    assert vectors[63] == [63, 1]
    assert client.embeddings.create.call_count == 2


def test_generation_request_is_stateless_and_grounded():
    client = Mock()
    client.responses.create.return_value.output_text = " Answer [1] "
    assert OpenAIAnswerer(client).answer("Question", "Evidence") == "Answer [1]"
    request = client.responses.create.call_args.kwargs
    assert request["store"] is False
    assert "only the supplied" in request["instructions"]
    assert "Evidence" in request["input"]
    assert "tools" not in request
    assert "previous_response_id" not in request


@pytest.mark.parametrize("adapter,method,args,resource", [
    (OpenAIEmbedder, "embed", (["text"],), "embeddings"),
    (OpenAIAnswerer, "answer", ("question", "context"), "responses"),
])
def test_api_errors_are_sanitized(adapter, method, args, resource):
    client = Mock()
    getattr(client, resource).create.side_effect = OpenAIError("sensitive upstream detail")
    with pytest.raises(RuntimeError) as error:
        getattr(adapter(client), method)(*args)
    assert "sensitive" not in str(error.value)


def test_empty_answer():
    client = Mock()
    client.responses.create.return_value.output_text = " "
    with pytest.raises(RuntimeError, match="empty"):
        OpenAIAnswerer(client).answer("question", "context")


def test_missing_key(monkeypatch):
    monkeypatch.setattr("src.llm.load_dotenv", lambda *args: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_client()


def test_streamlit_initial_render_and_blank_question():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert app.title[0].value == "🌱 Agri RAG Assistant"
    app.button[0].click().run()
    assert not app.exception
    assert app.warning[0].value == "Enter a question first."


def test_end_to_end_with_real_faiss_and_mocked_api(tmp_path):
    from src.ingest import load_documents
    from src.retriever import Retriever

    (tmp_path / "water.md").write_text("Record moisture before irrigation.")
    client = Mock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])
    client.responses.create.return_value.output_text = "Record moisture [1]."
    embedder = OpenAIEmbedder(client)
    retriever = Retriever.build(load_documents(tmp_path), embedder)
    retriever.save(tmp_path / "index")
    restored = Retriever.load(tmp_path / "index", embedder)
    result = RAGAssistant(restored, OpenAIAnswerer(client)).ask("What to record?")
    assert result.answer == "Record moisture [1]."
    assert result.sources == ["water.md"]
    assert "Record moisture before irrigation." in client.responses.create.call_args.kwargs["input"]
