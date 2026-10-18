# Demo 1 — RAG chatbot

**RAG = Retrieval Augmented Generation.** Instead of hoping the model memorised
your documents, you *find* the relevant passages yourself and paste them into
the prompt. The model's only job is to read them and answer.

```
            ┌──────────── ingest.py (run once) ─────────────┐
            │  data/*.md → split → embed → index/*.json     │
            └───────────────────────────────────────────────┘

            ┌──────────── chat.py (every question) ─────────┐
  question →│ embed → search index → paste into prompt →    │→ answer
            │ send to the chat model → cite sources         │
            └───────────────────────────────────────────────┘
```

## Run it

```bash
./run.sh ingest   # build the index   · or: ../.venv/bin/python ingest.py
./run.sh          # ask questions     · or: ../.venv/bin/python chat.py
./run.sh --ask "How many annual leave days do I get?"
```

| Flag | Effect |
|---|---|
| `--ask "..."` | One question, then exit — good for slides |
| `--k 5` | Retrieve 5 chunks instead of 3 |
| `--hide-prompt` | Don't print the full augmented prompt |

## The four stages, as printed on screen

1. **RETRIEVE** — the question is embedded with the *same* model used for the
   documents, then compared against every chunk with cosine similarity. The
   table shows rank, score, source file and a preview.
2. **AUGMENT** — the winning chunks are pasted into the prompt template. The
   full prompt is printed, so there is no mystery about what the model received.
3. **GENERATE** — the chat model (`gemini-3.5-flash-lite`) answers, with
   token counts and timing.
4. **CITE** — which file and which chunk each answer came from.

## Two models, two jobs

Students assume one model does everything. It does not:

- **embeddings** → `gemini-embedding-001` turns text into 3072 numbers
- **the answer** → `gemini-3.5-flash-lite` reads the retrieved text and writes

## Numbers to point at

- 4 documents → **15 chunks** at 450 characters with 80 characters of overlap
- Each chunk → a vector of **3072 numbers**
- A good match scores **~0.75**; an unrelated question tops out around **0.54**,
  which is why the code warns below 0.62
- An answer takes about **2 seconds** end to end

## No network? Switch the embeddings

```bash
EMBED_BACKEND=local ./run.sh ingest && EMBED_BACKEND=local ./run.sh
```

That uses the hashing keyword embedding in `shared/embeddings.py` — no key, no
quota, no install. It matches shared *words* rather than shared *meaning*, so
scores land lower (a good match ≈ 0.34) and an unrelated question scores a flat
**0.00**, which actually makes the guardrail demo even clearer.

## Teaching moments

**The guardrail.** Ask `Who won the 2022 football World Cup?`. The model knows
the answer, but the prompt says *answer ONLY from the context*, so it refuses.
Then open `chat.py`, delete the word `ONLY` from `PROMPT_TEMPLATE`, and ask
again. The guardrail collapses. Put it back.

**Chunk size matters.** Run `--k 1` and find a question that now fails, because
the single best chunk didn't contain the whole answer.

**It is just text in the end.** The "AI magic" is one f-string. Show
`PROMPT_TEMPLATE` at the top of `chat.py` and say so.

## Files

| File | Lines | Role |
|---|---|---|
| `data/*.md` | — | Fake Orbit Systems policies. Replace with your own and re-run ingest. |
| `ingest.py` | ~130 | Stage 1. Load, split, embed, save. |
| `vector_store.py` | ~90 | The vector database. Cosine similarity in numpy. |
| `chat.py` | ~180 | Stage 2. The four stages above. |
