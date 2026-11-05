# milvus-epub-summary

Chat with the content of an EPUB book. [`readepub.py`](readepub.py) extracts the
text of an EPUB, embeds it with an [Ollama](https://ollama.com) embedding model,
stores the vectors in [Milvus](https://milvus.io) (Milvus Lite, a local file),
and starts an interactive Ollama chat that retrieves relevant passages from
Milvus (RAG) to answer your questions.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Pull the models you want to use (defaults shown):

  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python readepub.py path/to/book.epub
```

On every run the Milvus database (`milvus_epub.db`) is cleaned up (all existing
collections are dropped), then the book is extracted, chunked, embedded and
stored. Then ask questions at the `You>` prompt; type `exit` or `quit` to leave.

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

## How it works

1. **Extract** – `ebooklib` reads each document in the EPUB; `BeautifulSoup` strips the HTML.
2. **Chunk** – text is split into ~1000-character overlapping chunks.
3. **Embed & store** – chunks are embedded via `ollama.embed` and inserted into a Milvus collection (cosine similarity), one collection per book.
4. **Chat** – each question is embedded, the top-k similar chunks are fetched from Milvus and given to the Ollama chat model as context.
