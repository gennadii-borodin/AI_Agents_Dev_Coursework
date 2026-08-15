from src.config import get_settings
from src.llm_provider import RouterAIProvider


def test_chat_completion_retries_on_empty(monkeypatch):
    provider = RouterAIProvider(get_settings())
    calls = {"n": 0}

    def fake_do_request(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            # emulate _do_request raising on empty provider content
            raise ValueError("LLM provider returned empty content")
        return '{"ok": true}', {}

    monkeypatch.setattr(provider, "_do_request", fake_do_request)
    out = provider.chat_completion(
        [{"role": "user", "content": "hi"}], max_tokens=10
    )
    assert out == '{"ok": true}'
    assert calls["n"] >= 2


def test_chat_completion_raises_after_empty_retries(monkeypatch):
    provider = RouterAIProvider(get_settings())
    calls = {"n": 0}

    def fake_do_request(payload):
        calls["n"] += 1
        raise ValueError("LLM provider returned empty content")

    monkeypatch.setattr(provider, "_do_request", fake_do_request)
    try:
        provider.chat_completion(
            [{"role": "user", "content": "hi"}], max_tokens=10
        )
        assert False, "expected ValueError after exhausting retries"
    except ValueError as e:
        assert "empty content" in str(e)
    assert calls["n"] >= 3
