"""STAGE 1 of RAG: turn a folder of documents into a searchable index.

Run this once, before the chatbot:

    python ingest.py

Pipeline:  load files → split into chunks → embed each chunk → save the index

Nothing is sent to the chat model here. This stage only uses the EMBEDDING
model, which turns text into a list of numbers that captures its meaning.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import config
from shared.embeddings import get_embedder
from shared.logger import get_logger
from vector_store import SimpleVectorStore

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
INDEX_PATH = HERE / "index" / "vector_index.json"

CHUNK_SIZE = 450        # characters per chunk
CHUNK_OVERLAP = 80      # characters repeated between neighbours, so sentences are not cut in half


def main() -> None:
    log = get_logger("RAG Ingest")
    log.banner(
        "RAG - STAGE 1: BUILDING THE INDEX",
        "load documents → split into chunks → embed → save",
    )
    config.require_embeddings()
    log.kv("Embedding backend", config.EMBED_BACKEND, "cyan")
    log.kv("Embedding model", config.embedding_label(), "cyan")

    # ---------------------------------------------------------------- STEP 1
    log.step("LOAD DOCUMENTS FROM DISK")
    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        log.error(f"No .md files found in {DATA_DIR}")
        raise SystemExit(1)

    documents: list[tuple[str, str]] = []   # (source filename, full text)
    for f in files:
        text = f.read_text(encoding="utf-8")
        documents.append((f.name, text))
        log.kv(f.name, f"{len(text):,} characters")
    log.ok(f"Loaded {len(documents)} documents")
    log.note("The model never sees these files directly - only the chunks we retrieve later.")

    # ---------------------------------------------------------------- STEP 2
    log.step("SPLIT DOCUMENTS INTO CHUNKS")
    log.kv("Chunk size", f"{CHUNK_SIZE} characters")
    log.kv("Chunk overlap", f"{CHUNK_OVERLAP} characters")
    log.note("Overlap means a sentence cut at a boundary still appears whole in a neighbour chunk.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )

    chunks: list[str] = []
    sources: list[str] = []
    for name, text in documents:
        pieces = splitter.split_text(text)
        chunks.extend(pieces)
        sources.extend([name] * len(pieces))
        log.kv(name, f"{len(pieces)} chunks")

    log.ok(f"{len(chunks)} chunks total")
    log.table(
        "First 3 chunks (what actually gets embedded)",
        ["#", "source", "chars", "preview"],
        [
            [i, sources[i], len(chunks[i]), chunks[i][:70].replace("\n", " ") + "…"]
            for i in range(min(3, len(chunks)))
        ],
    )

    # ---------------------------------------------------------------- STEP 3
    log.step("EMBED EVERY CHUNK (text → numbers)")
    embedder = get_embedder()

    t0 = time.perf_counter()
    with log.timed(f"Embedding {len(chunks)} chunks with {config.embedding_label()}"):
        vectors = embedder.embed_documents(chunks)
    embed_seconds = time.perf_counter() - t0

    dim = len(vectors[0])
    log.ok(f"Got {len(vectors)} vectors of {dim} dimensions each")
    log.kv("Example (chunk 0, first 5 numbers)", [round(x, 4) for x in vectors[0][:5]], "green")
    log.note(f"Two chunks about the same topic end up close together in this {dim}-dimension space.")

    # ---------------------------------------------------------------- STEP 4
    log.step("SAVE THE INDEX")
    store = SimpleVectorStore(backend=config.EMBED_BACKEND)
    store.add(chunks, sources, vectors)
    store.save(INDEX_PATH)
    size_mb = INDEX_PATH.stat().st_size / 1_000_000
    log.ok(f"Index saved to {INDEX_PATH}")
    log.kv("Index size on disk", f"{size_mb:.2f} MB")

    log.summary(
        {
            "Embedding backend": config.EMBED_BACKEND,
            "Documents": len(documents),
            "Chunks": len(chunks),
            "Vector dimensions": dim,
            "Embedding time": f"{embed_seconds:.2f}s",
            "Total time": f"{log.elapsed():.2f}s",
            "Next step": "python chat.py",
        }
    )


if __name__ == "__main__":
    main()
