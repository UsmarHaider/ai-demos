"""Chat model helpers shared by both projects.

Two things happen here:

1. `chat_model()` builds the Gemini chat client named in .env. It is wrapped in
   LangChain's interface, which is the whole reason LangChain exists - the rest
   of the lesson code never has to know which model is behind it, and swapping
   in another provider means editing this one function.

2. `invoke_with_retry()` survives rate limits instead of crashing. Hosted APIs
   return HTTP 429 when you go too fast; a real application waits and tries
   again, and we print it loudly so students see it happen.
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any, Sequence

from shared import config

# Some Gemini models use fixed sampling and ignore `temperature`. That is fine,
# but the warning clutters the screen during a live demo, so we hide it.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")


def chat_model(temperature: float = 0.0, tools: Sequence | None = None):
    """Build the chat model named in .env, optionally with tools bound to it."""
    config.require_chat()

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        temperature=temperature,
        timeout=180,
    )

    # Binding tools is what turns a chat model into an agent: it sends the tool
    # names, descriptions and argument schemas with every request.
    return llm.bind_tools(tools) if tools else llm


def _retry_seconds(message: str, default: float = 20.0) -> float:
    """When the server tells us how long to wait, use its number."""
    m = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 1, 65.0)
    m = re.search(r"seconds: (\d+)", message)
    if m:
        return min(float(m.group(1)) + 1, 65.0)
    return default


def invoke_with_retry(llm, payload: Any, log, max_attempts: int = 4):
    """llm.invoke(payload), but survives rate limits instead of crashing."""
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(payload)
        except Exception as exc:
            text = str(exc)
            is_rate_limit = (
                "429" in text
                or "ResourceExhausted" in type(exc).__name__
                or "quota" in text.lower()
                or "rate limit" in text.lower()
            )
            if not is_rate_limit or attempt == max_attempts:
                raise
            wait = _retry_seconds(text)
            log.warn(
                f"Rate limit hit (429) on attempt {attempt}/{max_attempts}. "
                f"Waiting {wait:.0f}s and retrying - this is normal on a busy or free endpoint."
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def message_text(message) -> str:
    """Get the plain text out of a model reply.

    Newer LangChain versions return `content` as a list of typed blocks
    (text, reasoning, signatures) instead of a plain string. This helper
    copes with both shapes so the lesson code stays readable.
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        if parts:
            return "".join(parts)

    # LangChain >= 1.0 also exposes a convenience property.
    text = getattr(message, "text", None)
    if callable(text):
        text = text()
    return text if isinstance(text, str) else str(content)
