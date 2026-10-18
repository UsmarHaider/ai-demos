# RAG & Agents — two teaching demos

Two small, fully-logged Python projects built for a live classroom session on
how LLM applications actually work.

| Folder | What it teaches |
|---|---|
| [`01_rag_chatbot/`](01_rag_chatbot/) | **RAG** — answering questions from *your documents* instead of the model's memory |
| [`02_agent_langgraph/`](02_agent_langgraph/) | **Agents** — a LangGraph chatbot that decides when to call tools, with two working tools |

Every step prints to the terminal in colour **and** is written to a log file in
`logs/`, so students see exactly what happened and you can scroll back afterwards.

Nothing here is a black box: the vector database is 90 lines of numpy you can
read in one sitting, and the agent graph is built node by node rather than with
a prebuilt helper.

> **No API keys are stored in this repository.** You supply your own in a local
> `.env` file, which is gitignored. See [step 3](#step-3--add-your-api-key).

---

## Contents

- [What you need first](#what-you-need-first)
- [Step 1 — Get the code](#step-1--get-the-code)
- [Step 2 — Install the dependencies](#step-2--install-the-dependencies)
- [Step 3 — Add your API key](#step-3--add-your-api-key)
- [Step 4 — Build the RAG index](#step-4--build-the-rag-index)
- [Step 5 — Run the pre-flight check](#step-5--run-the-pre-flight-check)
- [Step 6 — Demo 1: the RAG chatbot](#step-6--demo-1-the-rag-chatbot)
- [Step 7 — Demo 2: the LangGraph agent](#step-7--demo-2-the-langgraph-agent)
- [Which model does what](#which-model-does-what)
- [Running with no API key at all](#running-with-no-api-key-at-all)
- [Questions students will ask](#questions-students-will-ask)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Security notes](#security-notes)

---

## What you need first

| | |
|---|---|
| **Python 3.10 or newer** | `python3 --version` to check |
| **git** | to clone the repository |
| **A terminal** | macOS, Linux, or WSL on Windows |
| **A Gemini API key** | free, takes a minute to get — [step 3](#step-3--add-your-api-key) |

You do **not** need a GPU, a database, Docker, or a paid account.

---

## Step 1 — Get the code

```bash
git clone https://github.com/UsmarHaider/ai-demos.git
cd ai-demos
```

Every command from here on is run **from this folder**. That matters: the
scripts find `.env` and the virtualenv relative to it.

---

## Step 2 — Install the dependencies

```bash
./setup.sh
```

That creates a virtualenv at `.venv/` and installs the pinned versions from
`requirements.txt` into it. It takes a minute or two.

### Why a virtualenv, and not a plain `pip install`

This project needs `langchain-core` **1.x**. Many machines already have
`langchain-core` **0.3** installed globally for other work, and upgrading it
system-wide breaks those projects. So everything here lives in
`demo_project/.venv` and nothing outside this folder is touched.

**Consequence:** always launch through the `./run_*.sh` scripts, which pick the
right Python automatically. If you run a file by hand, name the venv's Python
explicitly:

```bash
cd 01_rag_chatbot && ../.venv/bin/python chat.py     # correct
cd 01_rag_chatbot && python3 chat.py                 # WRONG — ModuleNotFoundError
```

<details>
<summary>Doing it manually instead of with <code>setup.sh</code></summary>

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```
</details>

---

## Step 3 — Add your API key

Everything runs on Google's **Gemini** API. One key covers both demos, and the
free tier is enough for a whole class.

### 3a. Get a key

Go to **https://aistudio.google.com/apikey**, sign in with a Google account and
click *Create API key*. It is free and needs no card.

### 3b. Put it in a .env file

The repository ships `.env.example`, a template with **no real key in it**.
Copy it:

```bash
cp .env.example .env
```

Then open `.env` in any editor and replace the placeholder:

```ini
GOOGLE_API_KEY=paste-your-gemini-key-here
```

That is the only line you must change. The rest already has working defaults:

```ini
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
EMBED_BACKEND=gemini
EMBED_MODEL=models/gemini-embedding-001
```

### What every setting means

| Variable | Meaning |
|---|---|
| `GOOGLE_API_KEY` | Your Gemini key. Used for both chat and embeddings |
| `GEMINI_CHAT_MODEL` | Which Gemini model answers questions and drives the agent. `gemini-3.5-flash-lite` has the most generous free-tier limit |
| `EMBED_BACKEND` | `gemini` for real embeddings, `local` for the offline numpy fallback that needs no key |
| `EMBED_MODEL` | The Gemini embedding model. `models/gemini-embedding-001` gives 3072 dimensions |

⚠️ **`.env` is listed in `.gitignore`. Never commit it, and never paste a key
into a `.py` or `.md` file.** All code reads the key from the environment only —
`shared/config.py` is the single file that touches it, and it masks the key
before printing it.

---

## Step 4 — Build the RAG index

```bash
./run_ingest.sh
```

This reads the four documents, splits them into 15 chunks, embeds each one and
saves the result to `01_rag_chatbot/index/`. It takes about two seconds and
makes one API call.

**Do this before `check.sh`**, because the index is not in the repository — it
is generated, and `.gitignore` excludes it. A fresh clone has no index, and the
pre-flight check reports a red `FAIL` until you build one.

> You can skip this step if you like: `./run_rag.sh` builds the index
> automatically when it is missing. Run it by hand when you want a class to
> *watch* the index being built, or before `check.sh` so every line comes back
> green.

---

## Step 5 — Run the pre-flight check

```bash
./check.sh
```

This makes two real API calls and tells you in a few seconds whether everything
works. Run it before you walk into a classroom.

```
  PRE-FLIGHT CHECK — demo_project
  ──────────────────────────────────────────────────────────
[  OK  ] Running inside the project venv       ./.venv/bin/python3
[  OK  ] Packages import                       langchain-core 1.6.11
[  OK  ] Gemini API key present                gemini · key AQ.Ab8…wxyz
[  OK  ] Agent tools load                      get_current_datetime, calculator
[  OK  ] RAG index built                       15 chunks (gemini embeddings)
[  OK  ] Chat model responds                   gemini-3.5-flash-lite → ready
[  OK  ] Tool calling works                    get_current_datetime({...})
[  OK  ] Embeddings respond                    gemini-embedding-001 → 3072 dimensions
```

Every line is a separate failure you'd otherwise discover in front of an
audience. If one says `FAIL`, the message names the fix; see
[Troubleshooting](#troubleshooting).

---

## Step 6 — Demo 1: the RAG chatbot

### The problem it solves

An LLM only knows what it was trained on. It has never seen your company's
handbook. RAG lets it answer from *your* documents instead of its memory — and
cite which document it used.

### Run it

```bash
./run_rag.sh        # interactive
```

You built the index in [step 4](#step-4--build-the-rag-index); this is stage 2.

```bash
./run_rag.sh --ask "What is the per diem for international travel?"   # one question, no prompt
./run_rag.sh --k 5                                                    # retrieve 5 chunks, not 3
./run_rag.sh --hide-prompt                                            # cleaner screen
```

### How it works

It runs in two stages, and the split is the whole point:

| Stage | File | What happens |
|---|---|---|
| **Ingest** | `ingest.py` | load the documents → split into chunks → turn each chunk into a vector → save the index. Done **once**, offline |
| **Chat** | `chat.py` | embed the question → find the closest chunks → paste them into the prompt → generate → cite the source. Done **per question** |

The model is not being *taught* anything. Every question silently pastes the
relevant paragraphs into the prompt. RAG is retrieval plus prompt-building —
there is no training involved anywhere in this demo.

| File | Role |
|---|---|
| `data/*.md` | Four fictional company policy documents — the knowledge base |
| `ingest.py` | Stage 1, above |
| `vector_store.py` | A vector database in ~90 lines of numpy — no magic, nothing to install |
| `chat.py` | Stage 2, above |
| `index/` | The saved index. Generated, gitignored, safe to delete |

### Questions that demo well

| Question | Why it works in front of a room |
|---|---|
| `How many annual leave days do I get, and how many can I carry over?` | Two exact numbers from one document, similarity ≈ 0.75 |
| `What is the per diem for international travel?` | A different document — students watch the cited source change |
| `How long must my password be and how often do I change it?` | Two facts from a single chunk |
| `Who won the 2022 football World Cup?` | **The money shot.** The model knows this perfectly well, but answers *"I could not find that in the company documents."* RAG constrains it to the retrieved text — which is exactly what you want from a policy bot |

### Using your own documents

Drop `.md` or `.txt` files into `01_rag_chatbot/data/`, delete the old index,
and rebuild:

```bash
rm -rf 01_rag_chatbot/index && ./run_ingest.sh
```

---

## Step 7 — Demo 2: the LangGraph agent

### The problem it solves

The model can't *do* anything. It has no clock and it is unreliable at
arithmetic. An agent gives it tools and lets it decide when to reach for one.

### Run it

```bash
./run_agent.sh                                        # interactive, with memory
./run_agent.sh --ask "How many days are left in this year?"
./run_agent.sh --graph                                # just draw the graph and exit
```

In-chat commands: `/graph` `/tools` `/prompt` `/reset` `/quit`

### How it works

It is a **loop**, not a pipeline:

```
  user → [ MODEL ] → wants a tool? ──yes──→ [ TOOL NODE ] ──┐
                          │                                  │
                          no                                 └──► back to MODEL
                          ↓
                       answer
```

The graph is built explicitly with `StateGraph` rather than with a prebuilt
agent, because seeing the loop *is* the lesson.

**The two tools** (`02_agent_langgraph/tools.py`) are deliberately boring, so
the mechanism stays visible:

1. `get_current_datetime(timezone)` — the model has no clock, so it must ask.
2. `calculator(expression)` — safe arithmetic via an AST whitelist. It **never**
   calls `eval()`; anything outside the allowed node types is rejected.

### Questions that demo well — in this order

| Question | What it proves |
|---|---|
| `Hello, who are you?` | **Zero tools.** The model is *deciding*, not blindly calling |
| `What is the time in Tokyo right now?` | 1 tool, 2 loops. It extracts `Asia/Tokyo` from the word "Tokyo" by itself |
| `How many days are left in this year?` | **Best single demo** — chains *both* tools: datetime → calculator → answer, 3 loops |
| `What is 1575 divided by 3?` then `Now multiply that by 4` | Memory: "that" resolves to 525 from the previous turn |
| `What is 10 divided by zero?` | The tool returns `Error: division by zero.`, the model reads that and explains it in words — error handling, visible |
| `What time is it on Mars?` | **0 tools.** It answers from knowledge instead — a nice contrast with the Tokyo question |

### The correction students always need

**The model never runs the code.** It only *asks* for a tool by name with
arguments; your Python executes it and hands the result back. Say this while the
TOOL NODE step is on screen.

Then type `/tools`: it prints the tool descriptions sent to the model, generated
from the Python docstrings. The docstring *is* the prompt — that usually gets a
reaction.

---

## Which model does what

Two different models do two different jobs here. Worth saying out loud, because
students assume one model does everything.

| Job | Model | Dimensions / speed |
|---|---|---|
| Chat + agent reasoning | `gemini-3.5-flash-lite` | ~0.8–1.2 s per call |
| RAG embeddings | `gemini-embedding-001` | 3072 dimensions, ~0.5 s |

Neither can do the other's job. The embedding model never generates text — it
only produces coordinates. Ask a class what an embedding model would reply to
"hello"; the answer is that it has no reply.

### Swapping the model is a one-file change

`shared/llm.py` is one function, and it is the only place in the project that
knows which model is running. The other ~1000 lines never ask.

That is the entire reason the LangChain wrapper is there, and it is the lesson
worth drawing out: to move the chat side onto a different provider — or onto a
model you host yourself with Ollama, vLLM or LM Studio — you edit that one
function and change nothing else. Embeddings are the same story in
`shared/embeddings.py`.

Both demos print which model they used at the top of every run, so there is
never any doubt about what answered.

## Running with no API key at all

RAG embeddings can come from a backend that needs no network and no key:

```bash
EMBED_BACKEND=local ./run_ingest.sh     # rebuild the index
EMBED_BACKEND=local ./run_rag.sh        # ask questions
```

`local` is a hashing keyword embedding written in numpy (`shared/embeddings.py`).
It matches on shared **words** rather than shared **meaning**, so it is weaker —
but it has no key, no quota and nothing to install. The scores sit on a
different scale, which is itself worth showing students:

| Question | Gemini | local |
|---|---|---|
| A question the documents answer | ~0.75 | ~0.34 |
| An unrelated question | ~0.54 | **0.00** |

An index built with one backend cannot be searched with the other. The code
detects the mismatch and tells you to rebuild rather than returning nonsense.

You still need a chat key to *generate* an answer. Retrieval alone works offline.

---

## Questions students will ask

**"Why not just use ChatGPT?"** — The RAG demo answers from documents the model
has never seen, and cites them. Ask the World Cup question to show the
difference.

**"Does the model run the code?"** — No. It only *asks* for a tool by name with
arguments; your Python runs it. Say this while the TOOL NODE step is on screen.

**"What actually is a vector?"** — `./run_ingest.sh` prints the first five
numbers of a real 3072-dimension embedding. Point at them.

**"How does it know which tool to use?"** — Type `/tools` in the agent chat. The
descriptions it prints are generated from the function docstrings and are sent
to the model with every request.

**"Is this a real vector database?"** — Yes, just a small one.
`vector_store.py` is a list of chunks, a matrix of embeddings and cosine
similarity, in under 90 lines. Swapping in Chroma, pgvector, FAISS or Pinecone
means changing that one file; the `add` / `search` / `save` / `load` interface
stays the same.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | You used system `python` instead of the venv — use the `./run_*.sh` scripts, or `./.venv/bin/python` |
| `No such file or directory: ./setup.sh` | You are in the wrong folder, or need `chmod +x *.sh` |
| A key is reported missing | `.env` doesn't exist (`cp .env.example .env`), or you are not in the project folder |
| Model call hangs or times out | Run `./check.sh` — it makes a real call and reports what failed |
| `429 ... quota exceeded` | Already handled — the code waits and retries with a yellow warning. Keep talking; it recovers |
| Embeddings fail | `EMBED_BACKEND=local ./run_ingest.sh`, then `EMBED_BACKEND=local ./run_rag.sh` — no network needed |
| `No index found`, or `check.sh` says `RAG index built ... FAIL` | `./run_ingest.sh`. Expected on a fresh clone — the index is generated, not committed |
| `index built with 'X' but .env says 'Y'` | You changed `EMBED_BACKEND`. Run `./run_ingest.sh` to rebuild |
| Output too wide / wrapping | Shrink the terminal font, or pass `--hide-prompt` |

Gemini free-tier limits differ sharply by model, which bites during a live demo:

| Model | Requests/min | Daily cap |
|---|---|---|
| `gemini-3.5-flash-lite` | 25+ | generous — **use this one** |
| `gemini-3.8-flash` | 5 | generous |
| `gemini-2.5-flash-lite` | 10 | **20 per day** — avoid |

---

## Repository layout

```
.
├── README.md                  this file
├── PRESENTER_GUIDE.md         minute-by-minute script for a 60-minute session
├── PROMPTS.md                 the prompts inside the code, and build prompts
├── .env.example               template for your keys — copy to .env
├── requirements.txt           pinned versions, verified working together
├── setup.sh                   creates .venv and installs dependencies
├── check.sh                   pre-flight check — run before presenting
├── preflight.py               the nine checks it runs
├── run_ingest.sh              Demo 1, stage 1: build the RAG index
├── run_rag.sh                 Demo 1: the RAG chatbot
├── run_agent.sh               Demo 2: the LangGraph agent
│
├── shared/                    used by both demos
│   ├── config.py              reads .env, validates it, fails with a clear message
│   ├── llm.py                 one function, two providers
│   ├── embeddings.py          Gemini embeddings + the offline numpy fallback
│   └── logger.py              the colour terminal output and the log files
│
├── 01_rag_chatbot/
│   ├── data/*.md              four fictional policy documents
│   ├── ingest.py              load → split → embed → save
│   ├── vector_store.py        a vector database in ~90 lines of numpy
│   └── chat.py                retrieve → augment → generate → cite
│
├── 02_agent_langgraph/
│   ├── tools.py               get_current_datetime, calculator (AST whitelist)
│   └── agent.py               the StateGraph loop, built node by node
│
└── logs/                      generated at runtime, gitignored
```

### For presenters

- **`PRESENTER_GUIDE.md`** — a minute-by-minute script: RAG 25 min, agents
  25 min, "whose model is this anyway?" 4 min, "the prompts they should write"
  8 min.
- **`PROMPTS.md`** — every prompt used inside the code, plus the build prompts
  for recreating each project from scratch.

---

## Security notes

- **No credentials are in this repository.** Keys live only in your local
  `.env`, which `.gitignore` blocks. `.env.example` holds placeholders.
- **All code reads keys from the environment.** `shared/config.py` is the only
  place that touches them, and it masks the key when printing.
- **The calculator tool never calls `eval()`.** It parses the expression to an
  AST and walks it against a whitelist of node types, so `__import__` and
  attribute access are rejected rather than executed.
- **The policy documents are fictional.** "Orbit Systems" is invented; no real
  company data is in here.
- **A plain-HTTP model endpoint leaks your key.** Use TLS for anything beyond a
  private network.
- **Rotate a key the moment it is pasted anywhere public** — a chat window, a
  screenshot, a commit. Revoke at https://aistudio.google.com/apikey.

---

## Licence

Teaching material, free to reuse. Attribution welcome, not required.
