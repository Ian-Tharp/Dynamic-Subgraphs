"""Concrete tool runner coverage."""

from __future__ import annotations

import json

import pytest

from app.models import NodeKind
from app.runtime import (
    DuckDuckGoSearchProvider,
    TavilySearchProvider,
    build_default_search_provider,
    build_grounded_tool_runner,
    make_initial_state,
    tavily_search_tool,
    web_search_tool,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self, size: int = -1) -> bytes:
        del size
        return json.dumps(self._payload).encode("utf-8")


def test_web_search_tool_parses_duckduckgo_response(monkeypatch) -> None:
    # Ensure TAVILY_API_KEY isn't set, so we exercise the DDG fallback.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def fake_urlopen(request, *, timeout):
        assert timeout == 1.5
        assert "q=dynamic+subgraphs" in request.full_url
        return _FakeResponse(
            {
                "Heading": "Dynamic graph",
                "AbstractText": "A graph whose structure changes over time.",
                "AbstractURL": "https://example.test/dynamic-graph",
                "RelatedTopics": [
                    {
                        "Text": "Dynamic graph - example result",
                        "FirstURL": "https://example.test/result",
                    }
                ],
            }
        )

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    result = web_search_tool(
        {"query": "dynamic subgraphs", "limit": 3},
        timeout_seconds=1.5,
    )

    assert result["tool"] == "web_search"
    assert result["provider"] == "duckduckgo_instant_answer"
    assert result["query"] == "dynamic subgraphs"
    # Provider-synthesized summary lives in `answer` regardless of which
    # source (Abstract/Answer/Definition) supplied it.
    assert result["answer"] == "A graph whose structure changes over time."
    assert result["results"] == [
        {
            "title": "Dynamic graph",
            "url": "https://example.test/result",
            "snippet": "Dynamic graph - example result",
        }
    ]


def test_web_search_tool_falls_back_to_bing_html(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def fake_urlopen(request, *, timeout):
        del timeout
        if "api.duckduckgo.com" in request.full_url:
            return _FakeResponse(
                {
                    "Heading": "",
                    "AbstractText": "",
                    "Answer": "",
                    "RelatedTopics": [],
                }
            )
        assert "www.bing.com/search" in request.full_url
        html = """
        <html><body>
          <ol>
            <li class="b_algo">
              <h2><a href="https://example.test/docs">LangGraph docs</a></h2>
              <p>StateGraph compile returns a runnable graph.</p>
            </li>
          </ol>
        </body></html>
        """
        return _RawResponse(html.encode("utf-8"))

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    result = web_search_tool({"query": "langgraph compile"})

    assert result["provider"] == "bing_html_search"
    assert result["fallback_request_url"].startswith("https://www.bing.com/search?")
    assert result["results"] == [
        {
            "url": "https://example.test/docs",
            "title": "LangGraph docs",
            "snippet": "StateGraph compile returns a runnable graph.",
        }
    ]


# ---------- TavilySearchProvider ----------


def test_tavily_provider_posts_query_and_parses_response(monkeypatch) -> None:
    captured: dict = {}

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["method"] = request.get_method()
        return _FakeResponse(
            {
                "answer": "Dynamic subgraphs let you build adaptive workflows.",
                "results": [
                    {
                        "title": "Dynamic Subgraphs Overview",
                        "url": "https://example.test/intro",
                        "content": "A bounded runtime for transient workflow graphs.",
                        "score": 0.92,
                    },
                    {
                        "title": "Tavily Docs",
                        "url": "https://tavily.com/docs",
                        "content": "Search API for LLM agents.",
                        "score": 0.81,
                    },
                ],
            }
        )

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    provider = TavilySearchProvider("test-key", timeout_seconds=2.0)
    result = provider.search("dynamic subgraphs", limit=5)

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 2.0
    assert captured["body"]["api_key"] == "test-key"
    assert captured["body"]["query"] == "dynamic subgraphs"
    assert captured["body"]["max_results"] == 5
    assert captured["body"]["include_answer"] is True

    assert result["tool"] == "web_search"
    assert result["provider"] == "tavily"
    assert result["answer"] == "Dynamic subgraphs let you build adaptive workflows."
    assert len(result["results"]) == 2
    assert result["results"][0] == {
        "title": "Dynamic Subgraphs Overview",
        "url": "https://example.test/intro",
        "snippet": "A bounded runtime for transient workflow graphs.",
        "score": 0.92,
    }


def test_tavily_provider_respects_limit_when_api_returns_more(monkeypatch) -> None:
    def fake_urlopen(request, *, timeout):
        del request, timeout
        return _FakeResponse(
            {
                "answer": None,
                "results": [
                    {
                        "title": f"r{i}",
                        "url": f"https://x/{i}",
                        "content": "x",
                        "score": 0.5,
                    }
                    for i in range(10)
                ],
            }
        )

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    provider = TavilySearchProvider("key")
    result = provider.search("q", limit=3)

    assert len(result["results"]) == 3


def test_tavily_provider_raises_on_network_error(monkeypatch) -> None:
    def fake_urlopen(request, *, timeout):
        del request, timeout
        raise OSError("connection refused")

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    provider = TavilySearchProvider("key")

    with pytest.raises(RuntimeError, match="Tavily search failed"):
        provider.search("q", limit=3)


def test_tavily_provider_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="non-empty api_key"):
        TavilySearchProvider("")


def test_tavily_provider_rejects_empty_query() -> None:
    provider = TavilySearchProvider("key")
    with pytest.raises(ValueError, match="non-empty query"):
        provider.search("", limit=3)


# ---------- env-aware factory ----------


def test_default_provider_uses_tavily_when_api_key_set(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-from-env")

    provider = build_default_search_provider()

    assert isinstance(provider, TavilySearchProvider)


def test_default_provider_falls_back_to_duckduckgo_when_no_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    provider = build_default_search_provider()

    assert isinstance(provider, DuckDuckGoSearchProvider)


def test_default_provider_honors_explicit_api_key_argument() -> None:
    provider = build_default_search_provider(tavily_api_key="explicit-key")

    assert isinstance(provider, TavilySearchProvider)


def test_default_provider_can_be_forced_to_duckduckgo(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    provider = build_default_search_provider(prefer_tavily=False)

    assert isinstance(provider, DuckDuckGoSearchProvider)


# ---------- tavily_search_tool convenience ----------


def test_tavily_search_tool_raises_when_no_key_available(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        tavily_search_tool({"query": "x"})


def test_tavily_search_tool_uses_env_key_when_present(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")

    def fake_urlopen(request, *, timeout):
        del timeout
        body = json.loads(request.data.decode("utf-8"))
        assert body["api_key"] == "from-env"
        return _FakeResponse({"answer": None, "results": []})

    monkeypatch.setattr("app.runtime.tools.urllib.request.urlopen", fake_urlopen)

    result = tavily_search_tool({"query": "x"})

    assert result["provider"] == "tavily"


def test_grounded_tool_runner_executes_policy_lookup() -> None:
    runner = build_grounded_tool_runner()

    result = runner(
        make_initial_state(),
        {
            "tool_name": "policy_lookup",
            "args": {"query": "strict runners"},
        },
    )

    payload = result["result"]
    assert payload["tool"] == "policy_lookup"
    assert "web_search" in payload["allowed_tools"]
    assert "shell" in payload["forbidden_kinds"]
    assert NodeKind.TOOL_CALL.value == "tool_call"


class _RawResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _RawResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self, size: int = -1) -> bytes:
        del size
        return self._payload
