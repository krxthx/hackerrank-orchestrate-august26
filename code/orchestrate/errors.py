"""Centralized error types and classification -- the one place that turns a raw exception
(litellm error, tenacity RetryError, pydantic ValidationError, missing file, ...) into a
typed, meaningful message. Nothing else in this codebase should string-match error text or
format its own "something went wrong" message; call classify_llm_error / describe_parse_error
here instead, so every failure -- logged, checkpointed, or shown to a user -- explains what
happened and what to do about it, consistently.

Two kinds of failure in this codebase:
  - Per-row failures during a batch run (one message, one judge call). Expected and handled
    in place (fallback decision, skip, continue the batch) by pipeline.py / evaluate.py --
    but the *reason* recorded still comes from here, not an ad hoc f-string at the call site.
  - Fatal failures (bad config, missing dataset file, CLI misuse). These should stop the run
    immediately with one clear message, not a raw Python traceback -- see `fatal_error_boundary`.
"""

from __future__ import annotations

import contextlib
import logging
import sys

from tenacity import RetryError

logger = logging.getLogger(__name__)


class OrchestrateError(Exception):
    """Base class for every error this codebase raises deliberately. `user_message` is the
    short, meaningful explanation meant for a human (CLI stderr, transcript, reason field);
    str(exc) additionally includes the wrapped cause for logs.
    """

    def __init__(self, user_message: str, *, cause: BaseException | None = None):
        self.user_message = user_message
        self.cause = cause
        detail = f" (caused by: {cause})" if cause else ""
        super().__init__(f"{user_message}{detail}")


class ConfigError(OrchestrateError):
    """Missing/invalid environment configuration -- wrong or absent API key, unreachable
    endpoint, etc. Fixable by editing .env.
    """


class DatasetError(OrchestrateError):
    """A required dataset file is missing, empty, or malformed."""


class ContentFilterBlockedError(OrchestrateError):
    """An upstream provider/proxy safety filter rejected the request before a decision could
    be generated. Meaningful in this domain: the block is itself a phishing/scam signal, not
    a generic failure -- callers should route this to a mute/scam decision, not a blank fallback.
    """


class LLMCallError(OrchestrateError):
    """The LLM call failed after retries were exhausted, for a reason that isn't a content
    filter block (network, rate limit/quota, provider outage, unexpected response shape).
    """


class DecisionParseError(OrchestrateError):
    """A model's final answer (router or judge) could not be parsed into the expected schema."""


_CONTENT_BLOCK_SIGNALS = ("403", "forbidden", "content_filter", "content policy", "blocked")
_AUTH_SIGNALS = ("401", "unauthorized", "invalid api key", "authentication", "incorrect api key")
_RATE_LIMIT_SIGNALS = ("429", "rate limit", "resource_exhausted", "quota")


def root_cause(exc: BaseException) -> BaseException:
    """Unwrap tenacity's RetryError to the actual underlying exception it gave up on --
    RetryError's own message ("RetryError[<Future at ...>]") is never itself meaningful.
    """
    if isinstance(exc, RetryError) and exc.last_attempt is not None:
        inner = exc.last_attempt.exception()
        if inner is not None:
            return inner
    return exc


def classify_llm_error(exc: BaseException) -> OrchestrateError:
    """Turn a raw litellm/tenacity failure into a typed, meaningful error. This is the only
    place in the codebase that pattern-matches provider error text -- every call site gets a
    consistent, actionable message instead of re-deriving its own heuristic.
    """
    cause = root_cause(exc)
    text = str(cause).lower()

    if any(s in text for s in _CONTENT_BLOCK_SIGNALS):
        return ContentFilterBlockedError(
            "Upstream provider/proxy rejected the request via a safety/content filter before "
            "a decision could be generated.",
            cause=cause,
        )
    if any(s in text for s in _AUTH_SIGNALS):
        return ConfigError(
            "The configured LLM provider rejected the request's credentials -- check "
            "ORCHESTRATE_MODEL/ORCHESTRATE_JUDGE_MODEL and the matching *_API_KEY in .env.",
            cause=cause,
        )
    if any(s in text for s in _RATE_LIMIT_SIGNALS):
        return LLMCallError(
            "The LLM provider's rate limit or quota was exhausted -- increase "
            "ORCHESTRATE_LLM_PACING_SECONDS, switch models, or wait for the quota to reset.",
            cause=cause,
        )
    return LLMCallError(f"LLM call failed: {cause}", cause=cause)


def describe_parse_error(exc: Exception) -> str:
    """Consistent one-line wording for 'model output didn't match the expected schema',
    used by both the router's and the judge's output-parsing fallback paths.
    """
    return f"could not parse a valid response ({type(exc).__name__}: {exc})"


def describe_step_limit(max_steps: int) -> str:
    """Consistent wording for 'the agent hit its tool-call step ceiling without ever
    producing a final answer' -- kept distinct from describe_parse_error even though both
    currently land in the same kind of fallback decision, because the root cause (a stuck
    reasoning loop vs. a malformed final answer) is worth being honest about in the
    recorded reason rather than collapsing into one generic message.
    """
    return f"the agent exceeded its {max_steps}-step tool-call limit without reaching a final decision"


@contextlib.contextmanager
def fatal_error_boundary():
    """Wrap a CLI entry point's main(). Any OrchestrateError prints its clean user_message;
    any other uncaught exception prints a one-line summary (full traceback still goes to the
    logger) instead of dumping a raw stack trace on the user. This is the "global handler" --
    per-row/per-call failures inside a batch run should be caught and classified locally
    (see pipeline.run_pipeline, evaluate.run_judge_eval), not left to reach this boundary.
    """
    try:
        yield
    except OrchestrateError as exc:
        logger.error("fatal: %s", exc, exc_info=exc.cause or exc)
        print(f"\nFATAL: {exc.user_message}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # last-resort: users should never see a raw traceback
        logger.error("unexpected fatal error", exc_info=True)
        print(f"\nFATAL: unexpected error ({type(exc).__name__}: {exc}). Full traceback in the log.", file=sys.stderr)
        sys.exit(1)
