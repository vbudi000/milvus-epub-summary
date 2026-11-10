#!/usr/bin/env python3
"""Chat with the content of an EPUB using Milvus (vector store) and Ollama (LLM).

Flow:
  1. Extract text from the EPUB (per chapter).
  2. Split into overlapping chunks and embed them with an Ollama embedding model.
  3. Store the chunks in a local Milvus Lite database.
  4. Run an interactive chat; each question retrieves the most relevant chunks
     from Milvus and passes them to an Ollama chat model as context.
"""

import argparse
import hashlib
import re
import sys
import warnings
from pathlib import Path
from typing import cast

warnings.filterwarnings("ignore", category=UserWarning)

import ebooklib  # noqa: E402
import ollama  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from ebooklib import epub  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

DEFAULT_DB = "milvus_epub.db"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "llama3.2"
EMBED_BATCH = 32

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a book. "
    "Use the provided book excerpts as your primary source of knowledge. "
    "If the excerpts do not contain the answer, say so instead of guessing."
)


# --------------------------------------------------------------------------
# EPUB extraction and chunking
# --------------------------------------------------------------------------
def extract_epub(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return (book title, [(chapter title, chapter text), ...])."""
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    titles = book.get_metadata("DC", "title")
    book_title = titles[0][0] if titles else path.stem

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(" ", strip=True) if heading else item.get_name()
        text = soup.get_text("\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        if len(text) > 50:
            chapters.append((title, text))
    return book_title, chapters


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Pack paragraphs into ~size character chunks with a character overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pieces = []
    for p in paragraphs:  # hard-split very long paragraphs
        while len(p) > size:
            pieces.append(p[:size])
            p = p[size - overlap:]
        pieces.append(p)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current)
            current = current[-overlap:] + " " + piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return chunks


# --------------------------------------------------------------------------
# Milvus
# --------------------------------------------------------------------------
def collection_name_for(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    stem = re.sub(r"\W+", "_", path.stem)[:40].strip("_") or "book"
    return f"epub_{stem}_{digest}"


def embed(texts: list[str], model: str) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = ollama.embed(model=model, input=texts[i:i + EMBED_BATCH])
        vectors.extend(resp["embeddings"])
    return vectors


def ingest(client: MilvusClient, name: str, epub_path: Path, embed_model: str) -> None:
    print(f"Extracting {epub_path.name} ...")
    book_title, chapters = extract_epub(epub_path)
    records = []
    for chapter_title, text in chapters:
        for chunk in chunk_text(text):
            records.append({"chapter": chapter_title[:500], "text": chunk})
    summary_path = epub_path.parent / f"{epub_path.stem}_summary.md"
    if summary_path.is_file():
        summary_text = summary_path.read_text(encoding="utf-8")
        summary_chunks = chunk_text(summary_text)
        records.extend({"chapter": "Book summary", "text": c} for c in summary_chunks)
        print(f"Found {summary_path.name}: adding {len(summary_chunks)} summary chunks.")
    if not records:
        sys.exit("No text could be extracted from the EPUB.")
    print(f"'{book_title}': {len(chapters)} sections, {len(records)} chunks. Embedding ...")

    vectors = []
    for i in range(0, len(records), EMBED_BATCH):
        vectors.extend(embed([r["text"] for r in records[i:i + EMBED_BATCH]], embed_model))
        print(f"  {min(i + EMBED_BATCH, len(records))}/{len(records)}", end="\r")
    print()

    client.create_collection(
        collection_name=name,
        dimension=len(vectors[0]),
        metric_type="COSINE",
        auto_id=True,
    )
    rows = [{"vector": v, **r} for v, r in zip(vectors, records)]
    for i in range(0, len(rows), 500):
        client.insert(collection_name=name, data=rows[i:i + 500])
    print(f"Stored {len(rows)} chunks in Milvus collection '{name}'.")


def retrieve(client: MilvusClient, name: str, question: str, embed_model: str, top_k: int):
    qvec = embed([question], embed_model)[0]
    res = client.search(
        collection_name=name,
        data=[qvec],
        limit=top_k,
        output_fields=["chapter", "text"],
    )
    return [hit["entity"] for hit in res[0]]


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------
def chat_loop(client: MilvusClient, name: str, args: argparse.Namespace) -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"\nChatting with model '{args.chat_model}'. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        hits = retrieve(client, name, question, args.embed_model, args.top_k)
        context = "\n\n".join(f"[{h['chapter']}]\n{h['text']}" for h in hits)
        prompt = f"Book excerpts:\n{context}\n\nQuestion: {question}"

        print("Bot> ", end="", flush=True)
        reply = ""
        for part in ollama.chat(
            model=args.chat_model,
            messages=history + [{"role": "user", "content": prompt}],
            stream=True,
        ):
            token = part["message"]["content"]
            reply += token
            print(token, end="", flush=True)
        print("\n")

        # Keep history compact: store the plain question, not the retrieved context.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply})


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with an EPUB using Milvus and Ollama.")
    parser.add_argument("epub", type=Path, help="Path to the .epub file")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Milvus Lite db file (default: {DEFAULT_DB})")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL,
                        help=f"Ollama chat model (default: {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per question (default: 5)")
    args = parser.parse_args()

    if not args.epub.is_file():
        sys.exit(f"File not found: {args.epub}")

    client = MilvusClient(args.db)
    name = collection_name_for(args.epub)

    try:
        # Start every invocation from a clean database.
        for old in cast(list[str], client.list_collections()):
            client.drop_collection(old)
        ingest(client, name, args.epub, args.embed_model)
        chat_loop(client, name, args)
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error: {e}\nIs the model pulled? e.g. `ollama pull {args.embed_model}`")
    except ConnectionError:
        sys.exit("Cannot reach Ollama. Start it with `ollama serve`.")


if __name__ == "__main__":
    main()
