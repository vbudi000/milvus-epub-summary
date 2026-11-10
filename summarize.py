#!/usr/bin/env python3
"""Summarize chapters of an EPUB book using Ollama.

Flow:
  1. Parse the EPUB path from command-line arguments.
  2. Create the 'out' directory and a subfolder named after the EPUB file.
  3. Extract chapters from the EPUB.
  4. Summarize each chapter into a multi-bullet list using Ollama and write to a file.
"""

import argparse
import re
import shutil
import sys
from typing import Any
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*XMLParsedAsHTMLWarning.*")

import ebooklib  # noqa: E402
import ollama  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from ebooklib import epub  # noqa: E402

DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_OUT_DIR = "out"
DEFAULT_CHUNK_SIZE = 7900
DEFAULT_CHUNK_OVERLAP = 200

SYSTEM_PROMPT = (
    "You are an expert summarizer. Your task is to provide a comprehensive "
    "summary of the given book chapter, capturing all key events in "
    "a single paragraph of 4-6 sentences that only contains the story plot. "
    "Do not repeat the prompt, nor do they should specify the source."
)

SYSTEM_PROMPT_MAIN = (
    "You are an expert summarizer. Your task is to provide a comprehensive, "
    "summary of the whole book from the individual chapter summaries, "
    "capturing the important characters, key plot events, and the conclusion of the story."
)


def extract_epub(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Extract book title and list of (chapter_title, chapter_text) from an EPUB file."""
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as e:
        raise ValueError(f"Failed to read EPUB file '{path}': {e}") from e

    titles = book.get_metadata("DC", "title")
    book_title = titles[0][0] if titles else path.stem

    chapters: list[tuple[str, str]] = []
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


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in file and directory names."""
    clean = re.sub(r"[^\w\s-]", "", name).strip()
    return re.sub(r"[-\s]+", "_", clean)


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Pack paragraphs into ~size character chunks with a character overlap."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []
    pieces = []
    for p in paragraphs:
        while len(p) > size:
            pieces.append(p[:size])
            p = p[size - overlap:]
        pieces.append(p)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > size:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def summarize_chunk(
    chunk_content: str,
    context_desc: str,
    model: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Summarize an individual chunk using Ollama."""
    user_prompt = (
        f"{context_desc}\n\n"
        f"Summarize the main points and key events into a concise paragraphs from the following section:\n\n"
        f"{chunk_content}"
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"].strip()


def summarize_chapter(
    chapter_title: str,
    chapter_text: str,
    book_title: str,
    model: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Summarize a single chapter using chunking if content exceeds chunk size."""
    chunks = chunk_text(chapter_text, size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return ""

    if len(chunks) == 1:
        user_prompt = (
            f"Book: '{book_title}'\n"
            f"Chapter: '{chapter_title}'\n\n"
            f"Summarize the main plot lines in this chapter from the following content:\n\n"
            f"{chunks[0]}"
        )
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()

    # Multi-chunk chapter: summarize each chunk, then synthesize into final chapter summary
    chunk_summaries = []
    for c_idx, chunk in enumerate(chunks, start=1):
        context_desc = f"Book: '{book_title}' | Chapter: '{chapter_title}' (Part {c_idx}/{len(chunks)})"
        c_summary = summarize_chunk(chunk, context_desc, model, SYSTEM_PROMPT)
        chunk_summaries.append(c_summary)

    combined_chunk_summaries = "\n\n".join(chunk_summaries)
    combine_prompt = (
        f"Book: '{book_title}'\n"
        f"Chapter: '{chapter_title}'\n\n"
        f"Combine and summarize the following section summaries of this chapter into a single coherent summary:\n\n"
        f"{combined_chunk_summaries}"
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combine_prompt},
        ],
    )
    return response["message"]["content"].strip()


def summarize_epub(
    chapter_summary: str,
    book_title: str,
    model: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Summarize the chapter summaries to get a whole book summary using chunking if needed."""
    chunks = chunk_text(chapter_summary, size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return ""

    print(f"The number of the chunks: {len(chunks)}") 

    if len(chunks) == 1:
        user_prompt = (
            f"Book: '{book_title}'\n"
            f"Please provide a comprehensive summary of the following content:\n\n"
            f"{chunks[0]}"
        )
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_MAIN},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()

    # Multi-chunk: summarize each chunk of chapter summaries, then synthesize whole book summary
    chunk_summaries = []
    for c_idx, chunk in enumerate(chunks, start=1):
        print(f"Processing chunk # {c_idx}")
        context_desc = f"Book: '{book_title}' (Chapter Summaries Part {c_idx}/{len(chunks)})"
        c_summary = summarize_chunk(chunk, context_desc, model, SYSTEM_PROMPT)
        chunk_summaries.append(c_summary)

    combined_summaries = "\n\n".join(chunk_summaries)
    combine_prompt = (
        f"Book: '{book_title}'\n"
        f"Please provide a final comprehensive summary of the entire book from these section summaries:\n\n"
        f"{combined_summaries}"
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MAIN},
            {"role": "user", "content": combine_prompt},
        ],
    )
    return response["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and summarize an EPUB book using Ollama."
    )
    parser.add_argument("epub", type=Path, help="Path to the .epub file")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use for summarization (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk character size for long chapters and summaries (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap character size (default: {DEFAULT_CHUNK_OVERLAP})",
    )

    args = parser.parse_args()

    # 1. Validate EPUB argument
    if not args.epub.exists():
        raise FileNotFoundError(f"EPUB file not found: {args.epub}")
    if not args.epub.is_file():
        raise ValueError(f"Specified path is not a file: {args.epub}")
    if args.epub.suffix.lower() != ".epub":
        raise ValueError(f"File '{args.epub}' is not an EPUB file (expected .epub extension)")

    # 2. Create base output directory
    out_base = args.out_dir
    out_base.mkdir(parents=True, exist_ok=True)

    # 3. Create subdirectory named after the EPUB filename (stem), cleaning up before re-runs
    epub_subfolder = out_base / args.epub.stem
    if epub_subfolder.exists():
        shutil.rmtree(epub_subfolder)
    epub_subfolder.mkdir(parents=True, exist_ok=True)

    print(f"Reading EPUB: {args.epub}")
    print(f"Output directory: {epub_subfolder.resolve()}")

    # 4. Extract EPUB chapters
    book_title, chapters = extract_epub(args.epub)
    if not chapters:
        raise RuntimeError(f"No extractable chapters found in '{args.epub}'")

    print(f"Found {len(chapters)} chapter(s) for '{book_title}'.")
    print(f"Starting summarization using Ollama model '{args.model}'...\n")
    book_safe_title = sanitize_filename(book_title) or args.epub.stem
    output_file = epub_subfolder / f"{book_safe_title}_summary.txt"
    with output_file.open(mode="a", encoding="utf-8") as file:
        file.write(f"Book title: {book_title} \n\n")

        for idx, (chapter_title, chapter_text) in enumerate(chapters, start=1):
            print(f"[{idx}/{len(chapters)}] Summarizing: {chapter_title} ...", flush=True)

            try:
                summary = summarize_chapter(
                    chapter_title=chapter_title,
                    chapter_text=chapter_text,
                    book_title=book_title,
                    model=args.model,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            except ollama.ResponseError as e:
                sys.exit(f"Ollama error while summarizing '{chapter_title}': {e}\nIs the model pulled? e.g. `ollama pull {args.model}`")
            except ConnectionError:
                sys.exit("Cannot reach Ollama. Ensure Ollama is running (`ollama serve`).")

            file_content = (
                f"{summary}\n"
            )
            file.write(file_content)
            file.flush()

    print(f"\nAll {len(chapters)} chapter summaries saved to '{output_file}'.") 

    # 5. Summarize the whole epub

    book_summary_file = epub_subfolder / f"{book_safe_title}.txt"

    epub_summary_text = ""

    try:
        epub_summary_text = output_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        print("The file does not exist.")

    try:
        book_summary: str = summarize_epub(
            chapter_summary=epub_summary_text,
            book_title=book_title,
            model=args.model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except ollama.ResponseError as e:
        sys.exit(f"Ollama error while summarizing '{book_title}': {e}\nIs the model pulled? e.g. `ollama pull {args.model}`")
    except ConnectionError:
        sys.exit("Cannot reach Ollama. Ensure Ollama is running (`ollama serve`).")

    file_content = (
        f"Book: {book_title}\n"
        f"{'-' * 40}\n"
        f"Summary:\n{book_summary}\n\n"
    )

    book_summary_file.write_text(file_content)


if __name__ == "__main__":
    main()
