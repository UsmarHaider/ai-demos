# Demo 2 — LangGraph agent chatbot

A plain chatbot does `question → model → answer`.
An **agent** adds a loop: the model can stop and say *"I need a tool first"*.

```
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
```

The graph is built explicitly with `StateGraph` rather than a prebuilt agent,
because seeing the loop *is* the lesson.

## Which model is driving this?

`gemini-3.5-flash-lite`, Google's hosted API. Set `GEMINI_CHAT_MODEL` in
`.env` to use a different one — see `shared/llm.py`, which is the only file
that builds the model.

Tool calling is the part that matters here, and it works: the model returns a
proper `tool_calls` field, which is what LangGraph routes on. A full question
that chains two tool calls takes about **3 seconds**, split across three model
calls and two tool runs.

## Run it

```bash
./run.sh                       # or: ../.venv/bin/python agent.py
./run.sh --ask "What is the time in Tokyo right now?"
./run.sh --graph               # print the diagram and exit
```

Commands inside the chat: `/graph` `/tools` `/prompt` `/reset` `/quit`

## The two tools

| Tool | Why it exists |
|---|---|
| `get_current_datetime(timezone)` | The model has **no clock**. It cannot know the date. |
| `calculator(expression)` | The model **predicts text**, so arithmetic is unreliable. |

Both return *errors as strings* rather than raising, so the model can read the
error and deal with it. Try `What is 10 divided by zero?`: the tool returns
`Error: division by zero.`, and the model turns that into a proper explanation
instead of crashing.

`calculator` never uses `eval()`. It parses with `ast` and walks the tree with
an operator whitelist — worth 60 seconds on screen, because students will copy
whatever you show them.

## What the logs show, step by step

For every question:

- **AGENT NODE (loop N)** — how many messages are in context, which model, which
  tools are on offer, and the model's **DECISION**: call tools, or answer.
- **TOOL CALL** — the exact name and JSON arguments the model asked for. Stress
  that the model did *not* run anything; it only asked.
- **TOOL NODE** — our Python executing it, with the result and the milliseconds.
- **Router** — the conditional edge choosing `tools` or `END`.
- **RUN SUMMARY** — loops, LLM calls, tool calls, tools used, total time.

## Demo order that works

1. `Hello, who are you?` → **0 tools, 1 loop.** Establish that it *decides*.
2. `What is the time in Tokyo right now?` → **1 tool, 2 loops.** Note that
   nobody told it the string `Asia/Tokyo`; it worked that out from "Tokyo".
3. `How many days are left in this year?` → **2 tools, 3 loops.** It fetches the
   day of the year, then does the subtraction. This is the slide-worthy one.
4. `What is 1575 divided by 3?` then `Now multiply that by 4` → memory.
5. `/reset`, then ask the follow-up again → it has no idea what "that" means.
6. `What is 10 divided by zero?` → the tool errors, and the model recovers.

A good contrast to end on: `What time is it on Mars?` uses **no tool at all** —
the model answers from knowledge. Compare it with the Tokyo question and ask
the room why the two differ.

## Things students get wrong

**"The AI runs the code."** It does not. It emits a name and arguments. Your
`tool_node` runs the function. Everything else is message passing.

**"The docstring is a comment."** It is not — it is the prompt that tells the
model when to use the tool. Type `/tools` to show what the model actually sees.

**"More tools is better."** Every tool description is sent with every request,
so it costs tokens and dilutes the model's choice.

## Files

| File | Role |
|---|---|
| `tools.py` | The two tools, with the docstrings that make them usable |
| `agent.py` | State, the two nodes, the conditional edge, memory, logging, REPL |
