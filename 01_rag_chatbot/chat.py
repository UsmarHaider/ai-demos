"""STAGE 2 of RAG: answer questions using the index built by ingest.py.

    python chat.py                       # interactive
    python chat.py --ask "How many annual leave days do I get?"
    python chat.py --ask "..." --k 5     # retrieve more chunks
    python chat.py --hide-prompt         # don't print the full augmented prompt

For every question the program does FOUR things, and prints all four:

    1. RETRIEVE  - embed the question, find the closest chunks
    2. AUGMENT   - paste those chunks into a prompt template
    3. GENERATE  - send that prompt to the chat model
    4. CITE      - show which documents the answer came from

The model is told to answer ONLY from the retrieved context. That is the whole
point of RAG: the answer is grounded in your documents, not in the model's
memory, and you can prove it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import config
from shared.embeddings import get_embedder
from shared.llm import chat_model, invoke_with_retry, message_text
from shared.logger import get_logger
from vector_store import SimpleVectorStore

HERE = Path(__file__).resolve().parent
INDEX_PATH = HERE / "index" / "vector_index.json"

# A "this is probably not in the documents" warning threshold. Gemini
# embeddings score unrelated text around 0.54, so 0.62 is a sensible line.
# The local keyword backend scores unrelated text at 0.0, so its line is lower.
LOW_SCORE = {"gemini": 0.62, "local": 0.15}

# The prompt template is the heart of RAG. Show this slide to the students.
PROMPT_TEMPLATE = """You are the Orbit Systems policy assistant.

Answer the employee's question using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I could not find that in the company documents."
Do not use outside knowledge. Keep the answer under 80 words.
Quote the specific numbers or rules where they exist.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

QUESTION: {question}

ANSWER:"""


def answer_question(question: str, store, embedder, llm, log, k: int, show_prompt: bool) -> None:
    log.banner(f"QUESTION: {question}", "watch all four RAG stages below")

    # ---------------------------------------------------------- 1. RETRIEVE
    log.step("RETRIEVE - find the most relevant chunks")
    log.substep("Embedding the question with the same model used for the documents")
    t0 = time.perf_counter()
    q_vec = embedder.embed_query(question)
    log.kv("Question vector", f"{len(q_vec)} dimensions", "green")
    log.kv("Embedding time", f"{(time.perf_counter() - t0) * 1000:.0f} ms")

    log.substep(f"Comparing against {len(store)} stored chunks with cosine similarity")
    hits = store.search(q_vec, k=k)

    log.table(
        f"Top {k} chunks retrieved",
        ["rank", "score", "source", "preview"],
        [
            [i + 1, f"{h.score:.3f}", h.source, h.text[:60].replace("\n", " ") + "…"]
            for i, h in enumerate(hits)
        ],
    )
    log.note("Score is cosine similarity: 1.0 = identical meaning, 0.0 = unrelated.")

    if hits and hits[0].score < LOW_SCORE.get(config.EMBED_BACKEND, 0.5):
        log.warn("Top score is low - the documents probably do not cover this question.")

    # ----------------------------------------------------------- 2. AUGMENT
    log.step("AUGMENT - paste the chunks into the prompt template")
    context = "\n\n".join(f"[{h.source}]\n{h.text}" for h in hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    log.kv("Context characters", f"{len(context):,}")
    log.kv("Full prompt characters", f"{len(prompt):,}")

    # ---------------------------------------------------------- 3. GENERATE
    log.step("GENERATE - send the augmented prompt to the model")
    log.llm_request(config.chat_label(), prompt, show_prompt=show_prompt)

    t0 = time.perf_counter()
    response = invoke_with_retry(llm, prompt, log)
    seconds = time.perf_counter() - t0

    usage = getattr(response, "usage_metadata", None) or {}
    answer = message_text(response)
    log.llm_response(answer, seconds=seconds, usage=usage)
    log.answer(answer.strip())

    # -------------------------------------------------------------- 4. CITE
    log.step("CITE - where did this come from?")
    for i, h in enumerate(hits, 1):
        log.kv(f"{i}. {h.source}", f"chunk #{h.chunk_id} · score {h.score:.3f}", "green")

    log.summary(
        {
            "Question": question,
            "Chunks retrieved": len(hits),
            "Best match": f"{hits[0].source} ({hits[0].score:.3f})" if hits else "none",
            "LLM time": f"{seconds:.2f}s",
            "Tokens in / out": f"{usage.get('input_tokens', '?')} / {usage.get('output_tokens', '?')}",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chatbot over the Orbit Systems documents")
    parser.add_argument("--ask", help="ask one question and exit")
    parser.add_argument("--k", type=int, default=3, help="how many chunks to retrieve (default 3)")
    parser.add_argument("--hide-prompt", action="store_true", help="do not print the full prompt")
    args = parser.parse_args()

    log = get_logger("RAG Chat")
    config.require_chat()
    config.require_embeddings()

    if not INDEX_PATH.exists():
        log.error("No index found. Run this first:   python ingest.py")
        raise SystemExit(1)

    log.banner("RAG CHATBOT", "retrieve → augment → generate → cite")
    store = SimpleVectorStore.load(INDEX_PATH)
    log.kv("Index loaded", f"{len(store)} chunks")
    log.kv("Chat model", config.chat_label(), "cyan")
    log.kv("Embedding model", config.embedding_label(), "cyan")
    log.kv("Chunks per question (k)", args.k)

    # An index built with one embedding backend cannot be searched with
    # another - the numbers would mean completely different things.
    if store.backend not in (config.EMBED_BACKEND, "unknown"):
        log.error(
            f"This index was built with EMBED_BACKEND={store.backend}, but .env "
            f"now says {config.EMBED_BACKEND}. Rebuild it:   ./run_ingest.sh"
        )
        raise SystemExit(1)

    embedder = get_embedder()
    llm = chat_model(temperature=0)

    show_prompt = not args.hide_prompt

    if args.ask:
        answer_question(args.ask, store, embedder, llm, log, args.k, show_prompt)
        return

    log.console.print(
        "\n[bold]Ask a question about the Orbit Systems policies.[/] "
        "[dim]Type 'quit' to exit.[/]\n"
        "[dim]Try: How many annual leave days do I get?[/]\n"
        "[dim]Try: What is the per diem for international travel?[/]\n"
        "[dim]Try: Who won the 2022 football World Cup?  (to show the guardrail)[/]"
    )

    while True:
        try:
            question = log.console.input("\n[bold cyan]You ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer_question(question, store, embedder, llm, log, args.k, show_prompt)
        except Exception as exc:                       # keep the demo alive
            log.error(f"{type(exc).__name__}: {exc}")

    log.console.print("\n[dim]Bye. Full log: " + str(log.logfile) + "[/]")


if __name__ == "__main__":
    main()
