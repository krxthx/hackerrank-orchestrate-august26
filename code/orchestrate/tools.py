"""Tool registry. Decorate a function with @tool(...) and it becomes callable
by the agent loop, with the JSON schema auto-generated for the LLM.

Add real tools here once you know the challenge (e.g. a lookup against the
provided dataset, a calculator, a ticket classifier, etc).
"""

import inspect
from dataclasses import dataclass
from typing import Callable, get_type_hints

from orchestrate.types import ToolSpec

_PY_TO_JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def schema(self) -> dict:
        return self.spec.json_schema


REGISTRY: dict[str, Tool] = {}


def tool(description: str):
    def decorator(fn: Callable):
        hints = get_type_hints(fn)
        params = inspect.signature(fn).parameters
        properties = {}
        required = []
        for name, param in params.items():
            py_type = hints.get(name, str)
            properties[name] = {"type": _PY_TO_JSON_TYPE.get(py_type, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        schema = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        spec = ToolSpec(name=fn.__name__, description=description, json_schema=schema)
        REGISTRY[fn.__name__] = Tool(spec=spec, fn=fn)
        return fn

    return decorator


def all_schemas() -> list[dict]:
    return [t.schema for t in REGISTRY.values()]


def call(name: str, arguments: dict):
    if name not in REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return REGISTRY[name].fn(**arguments)


# --- example tool, delete/replace once the real challenge tools are known ---
@tool("Add two numbers together.")
def add(a: float, b: float) -> float:
    return a + b
