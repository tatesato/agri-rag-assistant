"""Verify development-only traceback output and credential redaction."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.diagnostics import safe_traceback


def test_chained_traceback_redacts_credentials(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-private-key')
    try:
        try:
            raise ValueError('test-private-key sk-proj-partial***redacted')
        except ValueError as cause:
            raise RuntimeError('Embedding request failed') from cause
    except RuntimeError as exc:
        result = safe_traceback(exc)
    assert 'ValueError' in result and 'RuntimeError' in result
    assert 'Traceback' in result
    assert 'test-private-key' not in result
    assert 'sk-proj' not in result


@pytest.mark.parametrize('mode,visible', [('development', True), ('production', False)])
def test_error_details_only_in_development(monkeypatch, mode, visible):
    monkeypatch.setenv('APP_ENV', mode)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-private-key')
    def fail():
        raise RuntimeError('diagnostic-marker test-private-key')
    monkeypatch.setattr('src.llm.create_client', fail)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
    next(b for b in app.button if b.label == '索引を作成・更新').click().run()
    assert not app.exception
    assert app.error
    assert bool(app.code) == visible
    if visible:
        assert 'diagnostic-marker' in app.code[0].value
        assert 'test-private-key' not in app.code[0].value
