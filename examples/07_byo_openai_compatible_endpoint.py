"""Bring-your-own OpenAI-compatible endpoint (vLLM, TGI, a gateway/proxy, ...).

`Model.openai_compatible(...)` targets any server speaking the OpenAI API.
This is a TEMPLATE — point base_url/api_key at your own endpoint. It builds the
engine and prints the resolved model ref; uncomment the run() call once your
endpoint is reachable.

Run:
    uv run python examples/07_byo_openai_compatible_endpoint.py
"""

from dotenv import load_dotenv

# Loads a local .env if present; otherwise set your keys in the environment
# (e.g. OPENAI_API_KEY). Works the same in-repo or standalone.
load_dotenv()

from dynamic_subgraphs import DynamicSubgraphs, EngineConfig, Model

model = Model.openai_compatible(
    "my-model",
    base_url="http://my-host:8000/v1",  # <- your endpoint
    api_key="not-needed",  # <- your key, if any
    # structured_method defaults to "json_schema" (safest for non-OpenAI servers)
)
engine = DynamicSubgraphs(EngineConfig(model=model))

print("provider:", model.provider)
print("model:", model.model)
print("base_url:", model.base_url)
print("structured_method:", model.structured_method)
print("engine ready:", isinstance(engine, DynamicSubgraphs))

# Once your endpoint is live:
# result = engine.run("Compare two options and recommend one.")
# print(result.response)
