"""Smoke test: exercises tool schema generation and the tool registry without
making a real LLM call (keeps CI/local runs free and fast).
"""

from orchestrate import tools


def test_add_tool_registered():
    assert "add" in tools.REGISTRY


def test_add_tool_call():
    assert tools.call("add", {"a": 2, "b": 3}) == 5


def test_schema_shape():
    schemas = tools.all_schemas()
    add_schema = next(s for s in schemas if s["function"]["name"] == "add")
    assert add_schema["function"]["parameters"]["required"] == ["a", "b"]
