"""An agentic chatbot built with LangGraph, with two tools and full logging.

    python agent.py                                  # interactive chat
    python agent.py --ask "What day is it in Tokyo?" # one question and exit
    python agent.py --graph                          # just draw the graph and exit

WHAT IS AN AGENT?
A normal chatbot does:            question → model → answer
An agent adds a loop:             question → model → "I need a tool" → run tool
                                              → model sees the result → answer

The model never runs code itself. It only *asks* for a tool by name, our
Python runs it, and we hand the result back. The loop below is the agent.

THE GRAPH
                      ┌───────────────┐
         START ──────▶│     agent     │  the model decides: answer, or call a tool?
                      └───────┬───────┘
                      ▲       │
       tool results   │       │ tool calls requested?
       go back in     │       ▼
                      │   ┌───────┐        no  ──▶  END  (final answer)
                      └───┤ tools │
                          └───────┘        yes ──▶  run the tools, loop again

We build this explicitly with StateGraph instead of using a prebuilt agent,
because the whole point of the lesson is to see the loop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from shared import config
from shared.llm import chat_model, invoke_with_retry, message_text
from shared.logger import get_logger
from tools import TOOLS

log = get_logger("Agent")

# The system prompt tells the model who it is and when to reach for a tool.
SYSTEM_PROMPT = """You are Orbit, a helpful assistant with access to tools.

Rules:
- You have no internal clock. For anything about today, now, the date or the
  time, you MUST call get_current_datetime. If the user does not name a city
  or country, use the timezone Asia/Karachi.
- You are unreliable at mental arithmetic. For ANY calculation, you MUST call
  calculator - even for simple sums.
- You may call several tools, one after another, to answer one question.
- After the tools return, answer the user in one or two short sentences.
- If a tool returns an error, read it, fix your arguments and try again once.
"""

MAX_ITERATIONS = 6          # a safety net so a confused model cannot loop forever
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Per-question counters, used for the summary table at the end of each turn.
STATS = {"iterations": 0, "tool_calls": 0, "llm_calls": 0, "tools_used": []}


# ===================================================================== STATE
class AgentState(TypedDict):
    """Everything the graph carries between nodes.

    `add_messages` means: when a node returns messages, APPEND them to the
    list rather than replacing it. That growing list is the agent's memory
    of this conversation.
    """
    messages: Annotated[list, add_messages]


# ===================================================================== NODES
def agent_node(state: AgentState) -> dict:
    """NODE 1 - ask the model what to do next."""
    STATS["iterations"] += 1
    STATS["llm_calls"] += 1
    iteration = STATS["iterations"]

    log.step(f"AGENT NODE (loop {iteration}) - asking the model what to do")
    messages = state["messages"]
    log.kv("Messages in context", len(messages))
    log.kv("Model", config.chat_label(), "cyan")
    log.kv("Tools offered", ", ".join(TOOLS_BY_NAME), "yellow")
    log.substep(f"Last message: {_describe(messages[-1])}")

    if iteration > MAX_ITERATIONS:
        log.error(f"Hit the {MAX_ITERATIONS} loop safety limit - stopping.")
        return {"messages": [AIMessage(content="I got stuck in a loop and stopped.")]}

    # The system prompt is prepended for the model only - we do not store it in
    # state, so the conversation history stays clean and easy to read.
    llm_messages = (
        messages
        if messages and isinstance(messages[0], SystemMessage)
        else [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    )

    t0 = time.perf_counter()
    response = invoke_with_retry(LLM, llm_messages, log)
    seconds = time.perf_counter() - t0

    usage = getattr(response, "usage_metadata", None) or {}
    log.llm_response(message_text(response), seconds=seconds, usage=usage)

    if response.tool_calls:
        log.ok(f"DECISION: call {len(response.tool_calls)} tool(s) before answering")
        for call in response.tool_calls:
            log.tool_call(call["name"], call["args"])
        log.note("The model did NOT run anything - it only asked. Our code runs it next.")
    else:
        log.ok("DECISION: no tool needed, answer the user directly")

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """NODE 2 - actually run the tools the model asked for."""
    last = state["messages"][-1]
    log.step(f"TOOL NODE - running {len(last.tool_calls)} tool(s) in Python")

    outputs = []
    for call in last.tool_calls:
        name, args = call["name"], call["args"]
        STATS["tool_calls"] += 1
        STATS["tools_used"].append(name)

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            result = f"Error: no tool named '{name}'."
            log.error(result)
            outputs.append(ToolMessage(content=result, tool_call_id=call["id"], name=name))
            continue

        log.substep(f"Executing {name} with arguments {args}")
        t0 = time.perf_counter()
        try:
            result = tool.invoke(args)
        except Exception as exc:
            result = f"Error running {name}: {exc}"
        seconds = time.perf_counter() - t0

        log.tool_result(name, result, seconds)
        outputs.append(ToolMessage(content=str(result), tool_call_id=call["id"], name=name))

    log.note("These results go back into the message list; the agent node runs again.")
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    """THE EDGE - after the agent node, loop to the tools or finish?"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        log.substep("Router: tool calls present → go to the TOOLS node")
        return "tools"
    log.substep("Router: no tool calls → go to END")
    return END


def _describe(message) -> str:
    kind = type(message).__name__
    text = message_text(message)[:70].replace("\n", " ")
    return f"{kind}: {text or '(empty - it was a tool request)'}"


# ================================================================= THE GRAPH
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")       # <- the loop

    # MemorySaver keeps the message list between turns, so the bot remembers
    # what you said earlier in the conversation.
    return builder.compile(checkpointer=MemorySaver())


GRAPH_PICTURE = """
                     ┌─────────────────┐
        START ──────▶│      agent      │  model decides: answer or use a tool?
                     └────────┬────────┘
                   ▲          │
    tool results   │          │  tool_calls?
    re-enter here  │          ├────────────── no ──▶  END   (final answer)
                   │          │
                   │          ▼ yes
                   │   ┌─────────────┐
                   └───┤    tools    │  our Python runs the function
                       └─────────────┘
"""


# ================================================================== RUN TURN
def run_turn(graph, question: str, thread_id: str) -> None:
    for key in STATS:
        STATS[key] = [] if isinstance(STATS[key], list) else 0

    log.banner(f"USER: {question}", "follow the agent loop below")
    t0 = time.perf_counter()

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    elapsed = time.perf_counter() - t0
    final = result["messages"][-1]
    log.answer(message_text(final).strip())

    log.summary(
        {
            "Agent loops": STATS["iterations"],
            "LLM calls": STATS["llm_calls"],
            "Tool calls": STATS["tool_calls"],
            "Tools used": ", ".join(STATS["tools_used"]) or "none",
            "Messages in memory": len(result["messages"]),
            "Total time": f"{elapsed:.2f}s",
        }
    )


# ====================================================================== MAIN
def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent with two tools")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--graph", action="store_true", help="print the graph and exit")
    args = parser.parse_args()

    if args.graph:
        print(GRAPH_PICTURE)
        return

    global LLM
    config.require_chat()

    log.banner("LANGGRAPH AGENT CHATBOT", "two tools · full logging · conversation memory")
    log.kv("Chat model", config.chat_label(), "cyan")
    log.kv("API key", config.masked_key(), "dim")
    log.kv("Tools", ", ".join(f"{t.name}" for t in TOOLS), "yellow")
    log.kv("Loop safety limit", MAX_ITERATIONS)

    log.step("WIRING THE GRAPH")
    LLM = chat_model(temperature=0, tools=TOOLS)
    graph = build_graph()
    log.ok("Graph compiled: START → agent ⇄ tools → END")
    log.console.print(GRAPH_PICTURE, style="cyan")

    log.step("TOOLS AVAILABLE TO THE MODEL")
    log.note("This description is generated from each function's signature and docstring.")
    for t in TOOLS:
        log.kv(t.name, t.description.strip().split("\n")[0], "yellow")
        log.console.print(f"      [dim]arguments: {list(t.args.keys())}[/]")

    thread_id = "classroom-demo"

    if args.ask:
        run_turn(graph, args.ask, thread_id)
        return

    log.console.print(
        "\n[bold]Chat with the agent.[/] [dim]Commands: /graph  /tools  /prompt  /reset  /quit[/]\n"
        "[dim]Try: What is the time in Tokyo right now?[/]\n"
        "[dim]Try: How many days are left in this year? (needs BOTH tools)[/]\n"
        "[dim]Try: Hello! (needs no tool at all)[/]"
    )

    turn = 0
    while True:
        try:
            text = log.console.input("\n[bold cyan]You ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue

        lowered = text.lower()
        if lowered in {"/quit", "quit", "exit", "/q"}:
            break
        if lowered == "/graph":
            log.console.print(GRAPH_PICTURE, style="cyan")
            continue
        if lowered == "/tools":
            for t in TOOLS:
                log.console.print(f"\n[bold yellow]{t.name}[/]  args={list(t.args.keys())}")
                log.console.print(f"[dim]{t.description}[/]")
            continue
        if lowered == "/prompt":
            log.code(SYSTEM_PROMPT, "text", title="system prompt")
            continue
        if lowered == "/reset":
            turn += 1
            thread_id = f"classroom-demo-{turn}"
            log.ok(f"Memory cleared. New conversation thread: {thread_id}")
            continue

        try:
            run_turn(graph, text, thread_id)
        except Exception as exc:                       # keep the demo alive
            log.error(f"{type(exc).__name__}: {exc}")

    log.console.print(f"\n[dim]Bye. Full log: {log.logfile}[/]")


LLM = None   # set in main()

if __name__ == "__main__":
    main()
