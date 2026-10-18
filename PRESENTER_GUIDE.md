# Presenter guide

About 65 minutes. Drop Part 3 and trim the exercises if you only have 45.

**Before you start:** `./check.sh` — a few seconds, and it tells you if anything
broke overnight. It checks that your key works and that tool calling still
works, which are the two things that would sink the session. Also make your
terminal font big and your window wide
(100 columns minimum, or the tables wrap).

---

## 0 · Setup on screen (2 min)

Open two terminal tabs, both in `demo_project`:

- Tab 1 → `./run_rag.sh`
- Tab 2 → `./run_agent.sh`

Also open `PROMPTS.md` in your editor — you will show prompts from it.

---

## Part 1 · RAG (25 min)

### 1.1 Set up the problem (3 min)

Ask the room: *"If I want a chatbot that knows my company's leave policy, what
are my options?"* Collect answers. Steer to the two real ones: fine-tune the
model (slow, expensive, stale the next day), or **give it the right page at the
moment you ask the question**. That second one is RAG.

### 1.2 Show the documents (2 min)

```bash
ls 01_rag_chatbot/data/
head -20 01_rag_chatbot/data/02_leave_policy.md
```

Say: *"The model has never seen these. I made them up this week."*

### 1.3 Build the index, live (5 min)

```bash
./run_ingest.sh
```

Walk through the four printed steps as they appear:

- **STEP 1** — 4 files loaded, character counts.
- **STEP 2** — 15 chunks. Explain chunk size 450 and overlap 80: *"the overlap
  is so a sentence cut in half still appears whole in the next chunk."*
- **STEP 3** — **stop here.** Point at `Example (chunk 0, first 5 numbers)`.
  That is an embedding: 3072 numbers describing the meaning of that chunk.
  Two chunks about leave end up near each other in that space.
  Mention that this step uses a *different model* (`gemini-embedding-001`) from
  the one that will write the answer (`gemini-3.5-flash-lite`).
- **STEP 4** — a 0.5 MB JSON file. *"That is our entire vector database."*

### 1.4 Ask a question (5 min)

```bash
./run_rag.sh --ask "How many annual leave days do I get, and how many can I carry over?"
```

Narrate the four stages:

- **RETRIEVE** — the question became a vector too; the table shows the three
  closest chunks and their scores. Best ≈ **0.75**.
- **AUGMENT** — *"this panel is the entire prompt. Nothing else is sent."*
  Read the first three lines of the template out loud.
- **GENERATE** — the answer in about 2 seconds, with token counts.
- **CITE** — the file and chunk the answer came from. *"This is why RAG gets
  used in companies: you can prove where the answer came from."*

### 1.5 The money shot (5 min)

```bash
./run_rag.sh --ask "Who won the 2022 football World Cup?"
```

The answer is *"I could not find that in the company documents."*

Ask: *"Does the model know who won? Of course it does."* Scroll up and show the
top score — about **0.54**, versus 0.75 for a real match. Then show the prompt
line that caused the refusal:

> Answer the employee's question using ONLY the context below.

**Optional live edit:** delete `ONLY` in `chat.py`, re-run, watch it answer
Argentina. Put it back. This lands harder than any slide.

### 1.6 Show the prompt (5 min)

Open `PROMPTS.md` → **Part A1**. Walk the table line by line: why the fixed
refusal string, why the delimiters, why the word limit.

---

## Part 2 · Agents (25 min)

### 2.1 The gap (3 min)

Ask the model a question it cannot possibly answer from training: *"What time is
it right now?"* It has no clock. Neither can it reliably do `1575 / 3`.

*"So we give it tools — and crucially, we let it decide when to use them."*

### 2.2 The graph (3 min)

```bash
./run_agent.sh --graph
```

Two boxes and one loop. `agent` decides; `tools` executes; the arrow back is
what makes it an agent instead of a chatbot.

### 2.3 No tool needed (2 min)

Start `./run_agent.sh`, then type:

```
Hello, who are you?
```

**1 loop, 0 tools.** The important observation: it *chose* not to use a tool.

### 2.4 One tool (5 min)

```
What is the time in Tokyo right now?
```

Pause on the **AGENT NODE** step: `DECISION: call 1 tool(s)`, then
`TOOL CALL get_current_datetime({"timezone": "Asia/Tokyo"})`.

Ask the room: *"Where did the string `Asia/Tokyo` come from? I never typed it."*
Answer: the model read the tool's docstring, which lists IANA names as examples,
and converted the word "Tokyo" itself.

Then the **TOOL NODE**: *"The model did not run this. My Python ran it, in one
millisecond, and handed the text back."*

### 2.5 Two tools chained (5 min)

```
How many days are left in this year?
```

**3 loops, 2 tools.** Datetime gives day 289 of the year; the calculator does
`365 - 289`; the model answers 76. Nobody wrote that plan — the model worked out
the sequence. Point at the RUN SUMMARY.

### 2.6 Memory (3 min)

```
What is 1575 divided by 3?
Now multiply that by 4
```

Note `Messages in memory` growing 4 → 8. Then `/reset` and ask
`Now multiply that by 4` again — it has no idea. That is `MemorySaver` and the
`thread_id`.

### 2.7 The docstring is the prompt (4 min)

```
/tools
```

Show that what the model sees is generated from the Python docstrings. Open
`PROMPTS.md` → **Part A3** and compare the good `calculator` docstring with
`"""Calculator."""`. Ask which one the model will use correctly.

Then `/prompt` — the system prompt. Every rule in it names a weakness of the
model and the tool that fixes it.

---

## Part 3 · Two models, two jobs (4 min)

Worth four minutes, because students assume one model does everything.

Open `shared/llm.py` and `shared/embeddings.py` side by side, then make these
points:

- **Two different models ran in the last 50 minutes.** `gemini-embedding-001`
  turned text into 3072 numbers; `gemini-3.5-flash-lite` wrote the sentences.
  Neither can do the other's job.
- **Embeddings never generate text.** Ask the room what an embedding model
  would reply to "hello". It has no reply — it only has coordinates.
- **The whole project knows about the model in exactly one place.** `llm.py` is
  one function. The other ~1000 lines never ask which model is running, which is
  the entire reason the LangChain wrapper is there.
- **So swapping the model is a one-file change.** To move the chat side onto a
  model you host yourself — Ollama, vLLM, LM Studio, anything speaking the
  OpenAI protocol — you edit that one function and nothing else. Say this
  plainly: it is what people actually mean by "switching to an open model".

Then show the key handling: `shared/config.py` is the only file that touches
`GOOGLE_API_KEY`, it reads it from `.env`, and it masks the key before printing
it. `.env` is gitignored. That is the habit you want them to copy.

## Part 4 · The prompts they should write (8 min)

Open `PROMPTS.md` → **Part B and C**. Make one point hard:

> These two projects were built by prompting, one file at a time, and reviewing
> each file before asking for the next.

Show Part B Prompt 3 (ingestion) and point at the specifics: exact chunk size,
exact model name, *"I am projecting this for students, so print each stage"*.
That last sentence is why the output looks the way it does.

Then **Part D**, the seven-point checklist, as the takeaway slide.

---

## Closing · Set the exercises (2 min)

From the end of `PROMPTS.md`:

1. Add a third tool. Write only the docstring first, and predict whether the
   model will call it.
2. Break the `calculator` docstring to one word. Which questions fail now?
3. Remove `ONLY` from the RAG prompt and explain what happens.
4. Point `data/` at your own notes and re-run `./run_ingest.sh`.
5. Switch `EMBED_BACKEND` to `local`, rebuild, and explain why the similarity
   scores all changed.

---

## If something breaks

| On screen | Say this, then do this |
|---|---|
| Yellow `Rate limit hit (429)` | *"The endpoint is busy — it is waiting and retrying, exactly like a real app should."* It recovers on its own. |
| `429 quota exceeded` repeatedly | Your model's per-minute cap. Switch `GEMINI_CHAT_MODEL` to `gemini-3.5-flash-lite` in `.env` — it has the most generous free-tier limit. |
| Embeddings fail | `EMBED_BACKEND=local ./run_ingest.sh`, then run the demo with the same prefix. No network needed. |
| A red error line | The chat loop survives errors. Just ask the next question. |
| Total freeze | Ctrl-C, re-run the `./run_*.sh` script. The RAG index is on disk, so nothing is lost. |
| Nothing works at all | Open `logs/` and talk through a log file from a previous run. Same material, no network. |

**Tip:** run each demo once before the session. The log files in `logs/` then
contain a complete transcript you can fall back on if the WiFi dies.
