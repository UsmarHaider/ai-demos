"""Pre-flight check. Run this before you walk into the classroom.

    .venv/bin/python -m shared.preflight

It makes exactly two API calls, then tells you whether the demo will work.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, FAIL, WARN = "\033[92m  OK  \033[0m", "\033[91m FAIL \033[0m", "\033[93m WARN \033[0m"
problems: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"[{OK}] {label:38s} {detail or ''}")
    except Exception as exc:
        print(f"[{FAIL}] {label:38s} {type(exc).__name__}: {exc}")
        problems.append(label)


print("\n  PRE-FLIGHT CHECK\n" + "  " + "─" * 58)


def _imports():
    import langgraph, numpy, rich                      # noqa: F401
    from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: F401
    import langchain_core
    return f"langchain-core {langchain_core.__version__}"


def _venv():
    if "/.venv/" not in sys.executable:
        raise RuntimeError(
            f"not using the project venv ({sys.executable})\n"
            "    use:  .venv/bin/python -m shared.preflight"
        )
    return sys.executable.replace(str(ROOT), ".")


def _key():
    from shared import config
    config.require_chat()
    return f"gemini · key {config.masked_key()}"


def _chat():
    from shared import config
    from shared.llm import chat_model, message_text
    reply = chat_model().invoke("Reply with the single word: ready")
    return f"{config.CHAT_MODEL} → {message_text(reply).strip()[:20]}"


def _toolcall():
    """The agent demo is worthless if the server cannot do tool calling."""
    from shared.llm import chat_model
    sys.path.insert(0, str(ROOT / "02_agent_langgraph"))
    from tools import TOOLS
    reply = chat_model(tools=TOOLS).invoke("What is the time in Tokyo right now?")
    if not reply.tool_calls:
        raise RuntimeError("model did not request a tool - the agent demo will not work")
    call = reply.tool_calls[0]
    return f"{call['name']}({call['args']})"


def _embed():
    from shared import config
    from shared.embeddings import get_embedder
    vec = get_embedder().embed_query("test")
    return f"{config.embedding_label()} → {len(vec)} dimensions"


def _index():
    path = ROOT / "01_rag_chatbot" / "index" / "vector_index.json"
    if not path.exists():
        raise RuntimeError("missing — run 01_rag_chatbot/run.sh ingest")
    import json
    from shared import config
    data = json.loads(path.read_text())
    backend = data.get("backend", "unknown")
    if backend not in (config.EMBED_BACKEND, "unknown"):
        raise RuntimeError(
            f"index built with '{backend}' but .env says '{config.EMBED_BACKEND}'"
            " — run 01_rag_chatbot/run.sh ingest"
        )
    return f"{len(data['texts'])} chunks ({backend} embeddings)"


def _tools():
    sys.path.insert(0, str(ROOT / "02_agent_langgraph"))
    from tools import TOOLS
    return ", ".join(t.name for t in TOOLS)


check("Running inside the project venv", _venv)
check("Packages import", _imports)
check("Gemini API key present", _key)
check("Agent tools load", _tools)
check("RAG index built", _index)
check("Chat model responds", _chat)
check("Tool calling works", _toolcall)
check("Embeddings respond", _embed)

print("  " + "─" * 58)
if problems:
    print(f"\n  \033[91m{len(problems)} problem(s):\033[0m " + ", ".join(problems) + "\n")
    sys.exit(1)
print("\n  \033[92mAll good. You are ready to present.\033[0m\n")
print("  Demo 1:  01_rag_chatbot/run.sh      Demo 2:  02_agent_langgraph/run.sh\n")
