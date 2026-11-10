# milvus-epub-summary

Chat with an EPUB book, with an LLM-generated summary included in the search index.

The project has two scripts that are meant to be run in order:

1. [`summarize.py`](summarize.py) – uses an [Ollama](https://ollama.com) model to
   summarize every chapter of an EPUB and then the whole book. The book summary is
   written next to the EPUB as `<epub name>_summary.md`.
2. [`readepub.py`](readepub.py) – extracts the EPUB text, embeds it with an Ollama
   embedding model, stores the vectors in [Milvus](https://milvus.io) (Milvus Lite,
   a local file) and starts an interactive chat that retrieves relevant passages
   (RAG). If `<epub name>_summary.md` exists next to the EPUB, it is **automatically
   loaded into Milvus as well**, so high-level questions ("What is the book about?",
   "How does it end?") are answered from the summary.

```
book.epub ──► summarize.py ──► book_summary.md ─┐
    │                                           ├──► readepub.py ──► Milvus ──► chat
    └───────────────────────────────────────────┘
```

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Pull the models you want to use (defaults shown):

  ```bash
  ollama pull qwen3:4b-instruct   # summarize.py
  ollama pull nomic-embed-text    # readepub.py (embeddings)
  ollama pull llama3.2            # readepub.py (chat)
  ```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Generate the book summary with `summarize.py`

```bash
python summarize.py path/to/book.epub
```

What it does:

1. Creates `out/<epub name>/` (any previous contents of that folder are deleted).
2. Extracts the chapters from the EPUB.
3. Summarizes each chapter with Ollama into a single paragraph and appends it to
   `out/<epub name>/<Book_Title>_summary.txt` (a per-chapter log you can inspect).
   Chapters longer than the chunk size are summarized in parts, then combined.
4. Summarizes all the chapter summaries into one whole-book summary (important
   characters, key events, conclusion).
5. Moves the result next to the EPUB as **`<epub name>_summary.md`**
   (e.g. `path/to/book.epub` → `path/to/book_summary.md`).

| Option               | Default           | Description                                              |
|----------------------|-------------------|----------------------------------------------------------|
| `--model NAME`       | `qwen3:4b-instruct` | Ollama model used for summarization                    |
| `--out-dir PATH`     | `out`             | Base directory for intermediate chapter summaries        |
| `--chunk-size N`     | `7900`            | Max characters per chunk sent to the model               |
| `--chunk-overlap N`  | `200`             | Characters of overlap between consecutive chunks         |

Example:

```bash
python summarize.py mybook.epub --model llama3.2 --chunk-size 6000
```

Summarizing runs one model call per chapter (more for long chapters), so it can
take a while for large books. If it stops with an Ollama error, check that
`ollama serve` is running and the model is pulled.

## Step 2: Load into Milvus and chat with `readepub.py`

Run it with the **same EPUB path** used in step 1, so the summary file is found:

```bash
python readepub.py path/to/book.epub
```

On every run the Milvus database (`milvus_epub.db`) is cleaned up (all existing
collections are dropped), then the book is extracted, chunked, embedded and
stored. If `path/to/book_summary.md` exists, its chunks are added to the same
collection under the chapter label `Book summary` and the console prints
`Found book_summary.md: adding N summary chunks.` If the file is missing, the
chat still works using only the book text.

Then ask questions at the `You>` prompt; type `exit` or `quit` to leave.

| Option                | Default             | Description                                  |
|-----------------------|---------------------|----------------------------------------------|
| `--chat-model NAME`   | `llama3.2`          | Ollama model used for chat                   |
| `--embed-model NAME`  | `nomic-embed-text`  | Ollama model used for embeddings             |
| `--db PATH`           | `milvus_epub.db`    | Milvus Lite database file                    |
| `--top-k N`           | `5`                 | Passages retrieved per question              |

Example:

```bash
python readepub.py mybook.epub --chat-model qwen3:4b-instruct --embed-model mxbai-embed-large
```

## Complete example

```bash
python summarize.py mybook.epub                # creates mybook_summary.md
python readepub.py mybook.epub                 # loads book + summary into Milvus, starts chat
```

## How it works

**`summarize.py`**

1. **Extract** – `ebooklib` reads each document in the EPUB; `BeautifulSoup` strips the HTML.
2. **Chunk** – chapter text is split into ~7900-character overlapping chunks.
3. **Summarize chapters** – each chapter (or each chunk, then combined) is summarized by Ollama.
4. **Summarize book** – the chapter summaries are summarized into one book summary and saved as `<epub name>_summary.md`.

**`readepub.py`**

1. **Extract** – same EPUB extraction as above.
2. **Chunk** – book text and the summary file are split into ~1000-character overlapping chunks.
3. **Embed & store** – chunks are embedded via `ollama.embed` and inserted into a Milvus collection (cosine similarity), one collection per book.
4. **Chat** – each question is embedded, the top-k similar chunks are fetched from Milvus and given to the Ollama chat model as context.
