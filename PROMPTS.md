# PROMPTS.md — the prompts to show the students

Two different kinds of prompt appear in this project, and students confuse them
constantly. Separate them clearly on stage:

- **Part A — prompts inside the code.** What *our program* sends to the model.
- **Part B / C — build prompts.** What *a developer* types into an AI coding
  assistant to generate the project in the first place.

---

# PART A — the prompts inside the code

## A1. The RAG prompt template (`01_rag_chatbot/chat.py`)

This is the single most important slide in the RAG lesson. Everything above it
in the pipeline exists only to fill in `{context}`.

```text
You are the Orbit Systems policy assistant.

Answer the employee's question using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I could not find that in the company documents."
Do not use outside knowledge. Keep the answer under 80 words.
Quote the specific numbers or rules where they exist.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

QUESTION: {question}

ANSWER:
```

**Point out, line by line:**

| Line | Why it is there |
|---|---|
| `using ONLY the context below` | This is what makes it RAG and not a chatbot |
| `reply exactly "I could not find..."` | A fixed refusal string is testable; "say you don't know" is not |
| `Do not use outside knowledge` | Belt and braces — without it the model leaks its training data |
| `--- CONTEXT START/END ---` | Clear delimiters stop the documents being read as instructions |
| `Keep the answer under 80 words` | Length control belongs in the prompt, not in your post-processing |
| `QUESTION:` … `ANSWER:` | The model completes the pattern you started |

**Live exercise:** delete the word `ONLY`, re-run the World Cup question, and
watch the guardrail collapse.

## A2. The agent system prompt (`02_agent_langgraph/agent.py`)

```text
You are Orbit, a helpful assistant with access to tools.

Rules:
- You have no internal clock. For anything about today, now, the date or the
  time, you MUST call get_current_datetime. If the user does not name a city
  or country, use the timezone Asia/Karachi.
- You are unreliable at mental arithmetic. For ANY calculation, you MUST call
  calculator - even for simple sums.
- You may call several tools, one after another, to answer one question.
- After the tools return, answer the user in one or two short sentences.
- If a tool returns an error, read it, fix your arguments and try again once.
```

**The teaching point:** every rule names a *weakness of the model* and the tool
that fixes it. That is how you write a system prompt for an agent — not
"you are a helpful assistant", but "here is what you are bad at, here is the
tool, here is when to reach for it".

## A3. The tool docstrings are prompts too (`02_agent_langgraph/tools.py`)

Students always miss this one. The docstring is **not a comment** — it is sent
to the model with every single request, and it is how the model decides which
tool to pick and what to put in the arguments.

```python
@tool
def calculator(expression: str) -> str:
    """Do exact arithmetic. Use this for ANY calculation, however simple.

    Language models predict text and are unreliable at mental arithmetic, so
    always use this tool instead of working the answer out yourself.

    Args:
        expression: A arithmetic expression in Python syntax, for example
            "1575 / 3 + 12", "22 - 7", "(4500 * 6) + 2000" or "2 ** 10".
            Numbers and the operators + - * / // % ** and brackets only.
            Do not include currency symbols, commas or units.
    """
```

Compare with the bad version and ask which one the model will use correctly:

```python
@tool
def calculator(expression: str) -> str:
    """Calculator."""          # the model has no idea when or how to call this
```

**Rule for students:** a tool description must say **what it does**, **when to
use it**, and **the exact format of every argument**, with examples.

---

# PART B — build prompts for Project 1 (RAG)

Type these one at a time into Claude Code / Cursor / ChatGPT. Do not paste all
five at once — the point is that each step is reviewed before the next.

### Prompt 1 — the skeleton

```text
Create a Python project folder called rag_chatbot that answers questions about
a folder of markdown documents.

Use langchain-google-genai for both the chat model and the embeddings.

Other libraries: langchain-text-splitters, numpy, rich, python-dotenv.
Read the API key and both model names from a .env file - never hard-code
them.

Structure it as two scripts I run separately:
  ingest.py  - builds a searchable index from the documents
  chat.py    - answers questions using that index

Create the folder structure and an empty version of each file with a docstring
explaining what it will do. Do not write the logic yet.
```

### Prompt 2 — the vector store

```text
Write vector_store.py: a minimal vector database in plain numpy, with no
external database.

class SimpleVectorStore with:
  add(texts, sources, embeddings)      store chunks and their vectors
  search(query_embedding, k=3)         return the k most similar chunks using
                                       cosine similarity, with their scores
  save(path) / load(path)              persist to a JSON file

Return search results as a dataclass with: text, source, chunk_id, score.
Comment the cosine similarity line - I am teaching this to students who have
not seen it before.
```

### Prompt 3 — ingestion

```text
Write ingest.py. It should:
1. read every .md file in data/
2. split them with RecursiveCharacterTextSplitter, chunk_size=450, overlap=80
3. embed every chunk with GoogleGenerativeAIEmbeddings using the model
   models/gemini-embedding-001
4. save the result with SimpleVectorStore

I am projecting this on a screen for students, so print each stage as a
numbered step with rich: how many files were loaded, how many chunks each file
produced, a preview of the first 3 chunks, the vector dimension, the first 5
numbers of one real embedding, and how long the embedding call took.
```

### Prompt 4 — the RAG chat loop

```text
Write chat.py. For each question:
1. RETRIEVE  embed the question, find the top k chunks, print a table of
             rank / similarity score / source file / preview
2. AUGMENT   paste the chunks into a prompt template and print the exact
             final prompt that will be sent
3. GENERATE  send it to the chat model and print the answer in a panel
4. CITE      list which file and chunk each piece of context came from

The prompt must tell the model to answer ONLY from the context, and to reply
exactly "I could not find that in the company documents." when the context
does not cover the question.

Support: interactive mode, --ask "question" for one shot, --k N, --hide-prompt.
```

### Prompt 5 — make it survive a live demo

```text
Make this robust enough to run in front of an audience:
- if the index file is missing, print how to build it instead of a stack trace
- if the API key is missing, print where to get one and exit cleanly
- if the API returns 429, wait the number of seconds the error asks for and
  retry up to 4 times, printing a visible warning
- in interactive mode, never let one failed question kill the session
- write every log line to a timestamped file in logs/ as well as the screen
```

---

# PART C — build prompts for Project 2 (LangGraph agent)

### Prompt 1 — the tools first

```text
Create tools.py with two LangChain tools using the @tool decorator.

1. get_current_datetime(timezone: str = "Asia/Karachi") -> str
   Returns the current date and time in that IANA timezone, including the day
   of the year. If the timezone name is invalid, return a helpful error STRING
   rather than raising, so the model can correct itself.

2. calculator(expression: str) -> str
   Evaluates an arithmetic expression. Do NOT use eval(). Parse with ast and
   allow only numbers and + - * / // % ** and brackets, via a whitelist.

Write long, explicit docstrings with argument examples - the model reads them
to decide when to call each tool.
```

### Prompt 2 — the graph

```text
Write agent.py using LangGraph. Build the graph explicitly with StateGraph -
do NOT use a prebuilt agent, because I am teaching the loop.

State: messages, using Annotated[list, add_messages]

Nodes:
  agent  - calls the chat model with the tools bound, returns its response
  tools  - runs whatever tools the model asked for, returns ToolMessages

Edges:
  START -> agent
  agent -> conditional: tool calls present? "tools" : END
  tools -> agent        (this edge is the loop)

Compile with a MemorySaver checkpointer and a thread_id so the chatbot
remembers earlier turns.
```

### Prompt 3 — the logging (this is the demo)

```text
Add logging so a student watching the terminal can narrate what the agent is
doing. For every question print, as numbered steps:

- which node is running and which loop number this is
- how many messages are in the context
- the model's DECISION: call tools, or answer directly
- each requested tool call with its name and arguments, as JSON
- each tool result with how many milliseconds it took
- the routing decision at the conditional edge
- the final answer in a green panel
- a summary table: loops, LLM calls, tool calls, tools used, total time

Use rich for colour. Also append everything to a log file.
```

### Prompt 4 — safety and polish

```text
Add to agent.py:
- a MAX_ITERATIONS limit of 6 so a confused model cannot loop forever
- a system prompt telling the model it has no clock and cannot do arithmetic,
  so it must use the tools
- retry with backoff when the API returns 429
- chat commands: /graph to print the graph, /tools to list the tool schemas,
  /prompt to show the system prompt, /reset to clear memory, /quit
- an --ask flag for one-shot questions
```

### Prompt 5 — prove it works

```text
Show me three example runs with the full log output:
1. a question needing one tool          ("what time is it in Tokyo?")
2. a question needing both tools        ("how many days are left in this year?")
3. a question needing no tool at all    ("hello, who are you?")

For each, confirm the loop count and which tools were called.
```

---

# PART D — what makes these prompts work

Give the students this checklist. Every prompt above follows it.

1. **State the stack.** Naming the libraries stops the assistant inventing its own.
2. **Describe the file layout before the logic.** Structure first, code second.
3. **One file per prompt.** Big prompts produce code you cannot review.
4. **Say who the output is for.** "I am projecting this for students" changed
   every print statement in this project.
5. **Give exact values.** `chunk_size=450, overlap=80` beats "chunk it sensibly".
6. **Specify the failure behaviour.** Most of Prompt 5 is about what happens
   when things go wrong — that is the difference between a toy and a demo that
   survives.
7. **Ask it to prove it.** Finish with "show me the output of a real run."

## Exercises to set

1. Add a third tool — a word counter, or a currency converter with fixed rates.
   Write only the docstring first, and predict whether the model will call it.
2. Break the `calculator` docstring down to one word. Which questions fail now?
3. In the RAG prompt, remove `ONLY`. Ask the World Cup question again.
4. Change `--k 3` to `--k 1`. Find a question that now gets the wrong answer,
   and explain why.
5. Point `data/` at your own notes or a past assignment brief and re-run `ingest.py`.
6. Set `EMBED_BACKEND=local` in `.env`, rebuild the index, and compare the
   similarity scores with the Gemini ones. Which questions still work, and why?
