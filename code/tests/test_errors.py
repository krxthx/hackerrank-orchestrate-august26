"""Offline tests for errors.py -- the single place error classification happens."""

import pytest
from tenacity import RetryError

from orchestrate.core.errors import (
    ConfigError,
    ContentFilterBlockedError,
    DatasetError,
    LLMCallError,
    OrchestrateError,
    classify_llm_error,
    describe_parse_error,
    describe_step_limit,
    fatal_error_boundary,
    root_cause,
)


class _FakeFuture:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def exception(self):
        return self._exc


def test_root_cause_unwraps_retry_error():
    inner = ValueError("boom")
    exc = RetryError(_FakeFuture(inner))
    assert root_cause(exc) is inner


def test_root_cause_passes_through_plain_exceptions():
    exc = ValueError("boom")
    assert root_cause(exc) is exc


def test_classify_llm_error_detects_content_block():
    exc = RetryError(_FakeFuture(Exception("APIError: OpenAIException - 403 Forbidden")))
    result = classify_llm_error(exc)
    assert isinstance(result, ContentFilterBlockedError)
    assert "safety/content filter" in result.user_message


def test_classify_llm_error_detects_auth_failure():
    exc = Exception("Error: Invalid API Key provided - Authentication failed")
    result = classify_llm_error(exc)
    assert isinstance(result, ConfigError)
    assert "API_KEY" in result.user_message


def test_classify_llm_error_detects_rate_limit():
    exc = Exception("RESOURCE_EXHAUSTED: 429 Too Many Requests, quota exceeded")
    result = classify_llm_error(exc)
    assert isinstance(result, LLMCallError)
    assert "rate limit" in result.user_message.lower() or "quota" in result.user_message.lower()


def test_classify_llm_error_falls_back_to_generic_llm_error():
    exc = Exception("Connection timed out")
    result = classify_llm_error(exc)
    assert isinstance(result, LLMCallError)
    assert not isinstance(result, ContentFilterBlockedError)
    assert not isinstance(result, ConfigError)


def test_describe_parse_error_includes_exception_type():
    msg = describe_parse_error(ValueError("no JSON object found"))
    assert "ValueError" in msg
    assert "no JSON object found" in msg


def test_describe_step_limit_includes_the_configured_ceiling():
    msg = describe_step_limit(6)
    assert "6-step" in msg
    assert "final decision" in msg


def test_orchestrate_error_str_includes_cause():
    cause = ValueError("underlying")
    exc = DatasetError("dataset file missing", cause=cause)
    assert "dataset file missing" in str(exc)
    assert "underlying" in str(exc)
    assert exc.user_message == "dataset file missing"


def test_fatal_error_boundary_prints_user_message_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        with fatal_error_boundary():
            raise DatasetError("dataset file missing")
    assert exc_info.value.code == 1
    assert "dataset file missing" in capsys.readouterr().err


def test_fatal_error_boundary_wraps_unexpected_exceptions(capsys):
    with pytest.raises(SystemExit) as exc_info:
        with fatal_error_boundary():
            raise ValueError("surprise")
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "unexpected error" in err
    assert "ValueError" in err


def test_fatal_error_boundary_lets_success_pass_through():
    with fatal_error_boundary():
        result = 1 + 1
    assert result == 2


def test_orchestrate_error_subclasses_are_catchable_generically():
    with pytest.raises(OrchestrateError):
        raise ContentFilterBlockedError("blocked")
