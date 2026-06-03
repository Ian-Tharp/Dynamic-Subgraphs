"""Concrete tool-call runners for allowlisted `tool_call` nodes.

The `web_search` tool routes through a `SearchProvider` abstraction so the
runtime can use a production-grade backend (Tavily) when configured and
gracefully fall back to a no-API-key path (DuckDuckGo Instant Answer →
Bing HTML scrape) otherwise.

For production deployments, set `TAVILY_API_KEY` in the environment. A
free tier (~1000 searches / month at time of writing) is sufficient for
development and small workloads. Get a key at https://tavily.com.

Output shape is uniform across providers — downstream LLM nodes consume
the same dict regardless of which backend ran the search:

    {
        "tool": "web_search",
        "provider": "tavily" | "duckduckgo_instant_answer" | "bing_html_search",
        "query": <str>,
        "answer": <str|None>,         # provider-synthesized summary if available
        "results": [{"title", "url", "snippet", "score"?}, ...],
        "request_url": <str|None>,    # for debugging only
        ...
    }
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any, Protocol

from app.models import DynamicRunState
from app.runtime.runners import NodeRunner, ToolCallable, run_tool_call

# ---------- search provider abstraction ----------


class SearchProvider(Protocol):
    """Anything that can run a web search and return a standardized dict."""

    def search(self, query: str, *, limit: int) -> dict[str, Any]: ...


class TavilySearchProvider:
    """Production search backend — Tavily's LLM-agent-focused API.

    Requires `api_key`. Returns relevance-scored snippets and an optional
    synthesized answer. Recommended for any deployment that does meaningful
    research work; the fallback `DuckDuckGoSearchProvider` is fine for
    development but quality and freshness suffer.
    """

    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 10.0,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("TavilySearchProvider requires a non-empty api_key")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._search_depth = search_depth
        self._include_answer = include_answer

    def search(self, query: str, *, limit: int) -> dict[str, Any]:
        if not query:
            raise ValueError("TavilySearchProvider.search requires a non-empty query")

        body = json.dumps(
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": limit,
                "search_depth": self._search_depth,
                "include_answer": self._include_answer,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self._ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "dynamic-subgraphs/0.1",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read(2_000_000)
        except Exception as exc:
            raise RuntimeError(
                f"Tavily search failed for query {query!r}: {exc}"
            ) from exc

        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Tavily search returned invalid JSON") from exc

        return {
            "tool": "web_search",
            "provider": "tavily",
            "query": query,
            "answer": payload.get("answer") or None,
            "results": [
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", ""))[:1500],
                    "score": item.get("score"),
                }
                for item in (payload.get("results") or [])
                if isinstance(item, dict)
            ][:limit],
            "request_url": self._ENDPOINT,
        }


class DuckDuckGoSearchProvider:
    """Fallback search backend — DuckDuckGo Instant Answer + Bing HTML scrape.

    No API key required. Quality is significantly lower than Tavily/Brave/
    Serper because the DDG Instant Answer endpoint returns mostly
    definitional content, and the Bing HTML fallback depends on Bing's
    page layout (fragile) and may violate Bing's TOS at scale.

    Use this only for development and demos. Configure `TAVILY_API_KEY`
    for any deployment that needs reliable search.
    """

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._timeout = timeout_seconds

    def search(self, query: str, *, limit: int) -> dict[str, Any]:
        if not query:
            raise ValueError(
                "DuckDuckGoSearchProvider.search requires a non-empty query"
            )

        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "dynamic-subgraphs/0.1"},
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read(1_000_000)
        except Exception as exc:
            raise RuntimeError(
                f"DuckDuckGo search failed for query {query!r}: {exc}"
            ) from exc

        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("DuckDuckGo search returned invalid JSON") from exc

        related = _flatten_related_topics(payload.get("RelatedTopics", []))
        ddg_results = [
            {
                "title": entry.get("text", "").split(" - ", 1)[0]
                if entry.get("text")
                else "",
                "url": entry.get("url", ""),
                "snippet": entry.get("text", ""),
            }
            for entry in related
        ][:limit]

        provider = "duckduckgo_instant_answer"
        fallback_request_url: str | None = None
        answer = (
            payload.get("Answer")
            or payload.get("AbstractText")
            or payload.get("Definition")
            or None
        )

        if not ddg_results and not answer:
            fallback_request_url, scraped = _bing_search(
                query, limit=limit, timeout_seconds=self._timeout
            )
            provider = "bing_html_search"
            ddg_results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in scraped
            ]

        return {
            "tool": "web_search",
            "provider": provider,
            "query": query,
            "answer": answer,
            "results": ddg_results,
            "request_url": url,
            "fallback_request_url": fallback_request_url,
            "heading": payload.get("Heading") or None,
            "abstract_url": payload.get("AbstractURL") or None,
        }


def build_default_search_provider(
    *,
    timeout_seconds: float = 10.0,
    prefer_tavily: bool = True,
    tavily_api_key: str | None = None,
) -> SearchProvider:
    """Pick the best available search provider from the environment.

    Returns `TavilySearchProvider` when an API key is available (either
    passed explicitly or via `TAVILY_API_KEY` env var) and
    `prefer_tavily=True`. Falls back to `DuckDuckGoSearchProvider` otherwise.
    """

    if prefer_tavily:
        key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if key:
            return TavilySearchProvider(key, timeout_seconds=timeout_seconds)
    return DuckDuckGoSearchProvider(timeout_seconds=timeout_seconds)


# ---------- tool callables (backed by providers) ----------


def make_web_search_tool(provider: SearchProvider) -> ToolCallable:
    """Wrap a `SearchProvider` as a `ToolCallable` for `tool_call` nodes."""

    def _tool(args: Mapping[str, Any]) -> dict[str, Any]:
        query = _query_from_args(args)
        limit = _int_arg(args, "limit", default=5, minimum=1, maximum=20)
        return provider.search(query, limit=limit)

    return _tool


# Backward-compat: keep the function signatures the existing tests use.


def web_search_tool(
    args: Mapping[str, Any],
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run a web search via the env-aware default provider.

    When `TAVILY_API_KEY` is set, uses Tavily. Otherwise uses DuckDuckGo +
    Bing HTML scrape (development fallback).
    """

    provider = build_default_search_provider(timeout_seconds=timeout_seconds)
    return make_web_search_tool(provider)(args)


def tavily_search_tool(
    args: Mapping[str, Any],
    *,
    api_key: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Run a web search explicitly through Tavily.

    Requires `api_key` or `TAVILY_API_KEY` in the environment. Raises if no
    key is available — useful when callers want to fail loudly rather than
    silently fall back to DuckDuckGo.
    """

    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "tavily_search_tool requires TAVILY_API_KEY (env var) or an "
            "explicit api_key argument. Get a free-tier key at https://tavily.com."
        )
    provider = TavilySearchProvider(key, timeout_seconds=timeout_seconds)
    return make_web_search_tool(provider)(args)


# ---------- grounded tool bundle ----------


def make_tool_call_runner(tools: Mapping[str, ToolCallable]) -> NodeRunner:
    """Build a `tool_call` runner backed by explicit tool implementations."""

    def _run_tool_call(
        state: DynamicRunState,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        return run_tool_call(state, params, tools=tools)

    return _run_tool_call


def build_grounded_tools(
    *,
    web_timeout_seconds: float = 10.0,
    search_provider: SearchProvider | None = None,
) -> dict[str, ToolCallable]:
    """Return concrete tool implementations for the default v1 allowlist.

    The `web_search` tool uses `search_provider` if supplied, otherwise the
    env-aware `build_default_search_provider()` (Tavily when configured,
    DuckDuckGo fallback otherwise).

    Honest naming caveats:
    - `web_search` is genuinely grounded (real network call).
    - `policy_lookup` is self-introspection of this runtime's allowlists,
      not external policy lookup.
    - `mock_document_extract` is string statistics, not real PDF/HTML extraction.
    - `create_follow_up_task` validates a task shape; it does NOT create
      anything (`created=False`, `side_effect=not_persisted`).
    """

    provider = search_provider or build_default_search_provider(
        timeout_seconds=web_timeout_seconds
    )
    return {
        "web_search": make_web_search_tool(provider),
        "policy_lookup": policy_lookup_tool,
        "create_follow_up_task": create_follow_up_task_tool,
        "mock_document_extract": document_extract_tool,
    }


def build_grounded_tool_runner(
    *,
    web_timeout_seconds: float = 10.0,
    search_provider: SearchProvider | None = None,
) -> NodeRunner:
    """Strict-mode-friendly runner for the allowlisted production tool bundle."""

    return make_tool_call_runner(
        build_grounded_tools(
            web_timeout_seconds=web_timeout_seconds,
            search_provider=search_provider,
        )
    )


# ---------- non-search tools ----------


def policy_lookup_tool(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return the local runtime policy relevant to tool and node execution."""

    from app.registry.allowlists import (
        DEFAULT_SUBAGENTS,
        DEFAULT_TOOLS,
        FORBIDDEN_KINDS,
    )

    return {
        "tool": "policy_lookup",
        "query": str(args.get("query") or args.get("q") or args.get("topic") or ""),
        "allowed_tools": sorted(DEFAULT_TOOLS),
        "allowed_subagents": sorted(DEFAULT_SUBAGENTS),
        "forbidden_kinds": sorted(FORBIDDEN_KINDS),
        "strict_runner_guidance": (
            "Use LangGraphExecutor(strict_runners=True) for production or "
            "evaluation runs so placeholder defaults cannot execute silently."
        ),
    }


def document_extract_tool(args: Mapping[str, Any]) -> dict[str, Any]:
    """Extract basic text statistics from caller-provided document content."""

    content = str(
        args.get("text")
        or args.get("content")
        or args.get("document")
        or args.get("item")
        or ""
    )
    words = content.split()
    preview_limit = _int_arg(
        args, "preview_chars", default=500, minimum=50, maximum=2000
    )
    return {
        "tool": "mock_document_extract",
        "char_count": len(content),
        "word_count": len(words),
        "preview": content[:preview_limit],
    }


def create_follow_up_task_tool(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return a structured follow-up task record without external side effects."""

    title = str(args.get("title") or args.get("task") or args.get("description") or "")
    description = str(args.get("description") or title)
    priority = str(args.get("priority") or "normal")
    return {
        "tool": "create_follow_up_task",
        "created": False,
        "side_effect": "not_persisted",
        "task": {
            "title": title or "Untitled follow-up",
            "description": description,
            "priority": priority,
        },
    }


# ---------- shared helpers ----------


def _query_from_args(args: Mapping[str, Any]) -> str:
    for key in ("query", "q", "search", "item"):
        value = args.get(key)
        if value:
            return str(value)
    if args:
        return json.dumps(dict(args), sort_keys=True, default=str)
    raise ValueError("web_search requires a non-empty query, q, search, or item arg")


def _int_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = args.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _flatten_related_topics(raw_topics: Any) -> list[dict[str, str]]:
    if not isinstance(raw_topics, list):
        return []

    results: list[dict[str, str]] = []
    for item in raw_topics:
        if not isinstance(item, dict):
            continue
        nested = item.get("Topics")
        if isinstance(nested, list):
            results.extend(_flatten_related_topics(nested))
            continue
        text = item.get("Text")
        url = item.get("FirstURL")
        if text or url:
            entry: dict[str, str] = {}
            if text:
                entry["text"] = str(text)
            if url:
                entry["url"] = str(url)
            results.append(entry)
    return results


def _bing_search(
    query: str,
    *,
    limit: int,
    timeout_seconds: float,
) -> tuple[str, list[dict[str, str]]]:
    params = urllib.parse.urlencode({"q": query})
    url = f"https://www.bing.com/search?{params}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 dynamic-subgraphs/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(2_000_000)
    except Exception as exc:
        raise RuntimeError(
            f"bing search fallback failed for query {query!r}: {exc}"
        ) from exc

    parser = _BingResultsParser(limit=limit)
    parser.feed(raw.decode("utf-8", errors="replace"))
    return url, parser.results


class _BingResultsParser(HTMLParser):
    def __init__(self, *, limit: int) -> None:
        super().__init__(convert_charrefs=True)
        self._limit = limit
        self.results: list[dict[str, str]] = []
        self._in_result = False
        self._depth = 0
        self._in_title = False
        self._in_snippet = False
        self._current: dict[str, str] = {}
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "li" and "b_algo" in classes and len(self.results) < self._limit:
            self._in_result = True
            self._depth = 1
            self._current = {}
            self._title_parts = []
            self._snippet_parts = []
            return

        if not self._in_result:
            return

        self._depth += 1
        if tag == "h2":
            self._in_title = True
        elif tag == "a" and self._in_title and "url" not in self._current:
            href = attr_map.get("href")
            if href:
                self._current["url"] = _clean_bing_url(href)
        elif tag == "p" and not self._current.get("snippet"):
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if not self._in_result:
            return

        if tag == "h2":
            self._in_title = False
            self._current["title"] = _normalize_text("".join(self._title_parts))
        elif tag == "p" and self._in_snippet:
            self._in_snippet = False
            self._current["snippet"] = _normalize_text("".join(self._snippet_parts))

        self._depth -= 1
        if self._depth <= 0:
            if self._current.get("title") or self._current.get("url"):
                self.results.append(dict(self._current))
            self._in_result = False
            self._current = {}

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)


def _clean_bing_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    encoded = urllib.parse.parse_qs(parsed.query).get("u", [None])[0]
    if encoded and encoded.startswith("a1"):
        payload = encoded[2:]
        padded = payload + "=" * ((4 - len(payload) % 4) % 4)
        try:
            return base64.urlsafe_b64decode(padded).decode("utf-8")
        except Exception:
            return url
    return url


def _normalize_text(value: str) -> str:
    return " ".join(value.split())
