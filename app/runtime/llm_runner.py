"""Real-LLM runner for `NodeKind.LLM_CALL` nodes.

The default `run_llm_call` in `runners.py` just echoes the instruction; it
exists so the rest of the pipeline can be exercised without spending tokens.
This module wires a real chat model.

Design notes:

- The runner depends on `BaseChatModel`, not on any specific provider.
  `build_openai_llm_runner` is a compatibility convenience factory; SDK
  assembly can pass any provider-built chat model to `ChatLlmRunner`.
- Upstream state values are injected into the user message so the model
  can see what previous nodes produced. This is how a downstream node like
  `compare_and_recommend` actually has the summaries to compare.
- The runner returns `{"result": <text>}`, matching the existing contract
  the wrapper expects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.models import DynamicRunState
from app.runtime.runners import run_reduce
from app.runtime.state import render_value_for_prompt

_SYSTEM_PROMPT = (
    "You are one node in a larger workflow graph. Follow the instruction "
    "below precisely. If 'Upstream state' is provided, treat those values "
    "as ground truth produced by previous nodes; use them to inform your "
    "response. Respond with the output content only — no preamble, no "
    "narration, no JSON wrapping."
)


class ChatLlmRunner:
    """Concrete `NodeRunner` for `llm_call` nodes, backed by any chat model."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat = chat_model

    def __call__(
        self,
        state: DynamicRunState,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        instruction = str(params["instruction"])
        upstream = state.get("values", {}) if state else {}

        user_parts: list[str] = [f"Instruction:\n{instruction}"]
        if upstream:
            rendered = "\n".join(
                f"- {key}: {render_value_for_prompt(value)}"
                for key, value in sorted(upstream.items())
            )
            user_parts.append(f"\nUpstream state:\n{rendered}")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(user_parts)),
        ]

        response = self._chat.invoke(messages)
        text = getattr(response, "content", None)
        if not isinstance(text, str):
            text = str(text)
        return {"result": text}


OpenAILlmRunner = ChatLlmRunner


def build_openai_llm_runner(
    model: str = "gpt-5.4-nano",
    *,
    temperature: float | None = None,
) -> ChatLlmRunner:
    """Convenience factory: build a `ChatLlmRunner` backed by ChatOpenAI.

    Requires `OPENAI_API_KEY` in the environment. The langchain-openai import
    is local so callers that don't use a real LLM don't pull the dependency.
    """

    from app.runtime.chat_models import build_openai_chat

    return ChatLlmRunner(build_openai_chat(model, temperature=temperature))


_REDUCE_SYSTEM_PROMPT = (
    "You are a reducer node in a workflow graph. You will be given an "
    "instruction and a set of named inputs produced by upstream nodes. "
    "Synthesize a single response that follows the instruction precisely "
    "using only those inputs. Respond with the synthesized content only — "
    "no preamble, no narration, no JSON wrapping."
)


class LlmReduceRunner:
    """Reduce runner that supports `llm_summarize` via a chat model.

    For `strategy in {"concat", "merge_dict"}` it delegates to the
    deterministic `run_reduce`. For `strategy == "llm_summarize"` it sends
    the declared `input_keys` (rendered from state) plus the params'
    `instruction` to the model and routes the response into `output_key`.
    """

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat = chat_model

    def __call__(
        self,
        state: DynamicRunState,
        params: Any,
    ) -> dict[str, Any]:
        strategy = params.get("strategy", "concat")
        if strategy != "llm_summarize":
            return run_reduce(state, params)

        values = state.get("values", {}) if state else {}
        input_keys = list(params["input_keys"])
        output_key = str(params["output_key"])
        instruction = str(params["instruction"])

        rendered = "\n".join(
            f"- {key}:\n{render_value_for_prompt(values[key])}" for key in input_keys
        )
        messages = [
            SystemMessage(content=_REDUCE_SYSTEM_PROMPT),
            HumanMessage(content=f"Instruction:\n{instruction}\n\nInputs:\n{rendered}"),
        ]

        response = self._chat.invoke(messages)
        text = getattr(response, "content", None)
        if not isinstance(text, str):
            text = str(text)
        return {output_key: text}


def build_openai_reduce_runner(
    model: str = "gpt-5.4-nano",
    *,
    temperature: float | None = None,
) -> LlmReduceRunner:
    """Convenience factory: `LlmReduceRunner` backed by ChatOpenAI."""

    from app.runtime.chat_models import build_openai_chat

    return LlmReduceRunner(build_openai_chat(model, temperature=temperature))
