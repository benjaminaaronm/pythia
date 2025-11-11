# pythia

`pythia` is a small Retrieval-Augmented Generation (RAG) system built in Python as part of a machine learning engineer assignment.

It answers business-style questions using a curated subset of Wikipedia articles (banking, consulting, data science, machine learning, deep learning, LLMs, telecom, energy, risk, etc.).

---

## 1. Models and architecture (high level)

The pipeline uses:

- **Document store:** plain text `.txt` files from Wikipedia-style articles.
- **Vector store:** a JSON file containing chunk texts, metadata, and embeddings.
- **Retriever–ranker:**
  - BM25 lexical retrieval (`rank_bm25`) to preselect candidate chunks.
  - Embedding-based cosine similarity for re-ranking.
- **LLM:** TinyLlama via Ollama to generate the final answer from retrieved context.

### Models (served by [Ollama](https://ollama.com/))

Configured in `pythia/config.py`:

- **Embedding model**

  ```python
  EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
  ```

- **LLM**

  ```python
  LL_MODEL = "tinyllama"
  ```

These are pulled and served locally by Ollama. All embedding and generation calls go through the `ollama` Python client.

---

## 2. Prerequisites

You will need:

- **Python**

  The project has been built with Python 3.12.

- **Ollama**

  - Install from: https://ollama.com/download  
  - Make sure the Ollama service is running (it listens on `http://127.0.0.1:11434` by default).

- (Optional) **Docker**  
  Only needed if you want to run the API in a container.

---

## 3. Set up Ollama

After installing and starting Ollama, pull the two models used by pythia:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull tinyllama
```

---

## 4. Python environment

From the project root:

```bash
python -m venv .venv

# PowerShell / CMD:
.venv\Scripts\activate

# (On Linux/macOS, use: source .venv/bin/activate)

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Data and index

The repository contains:

- Raw text files (Wikipedia-style) under `data/raw/wikipedia/`.
- A **prebuilt index** in `data/processed/pythia_index.json` with embeddings created using `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` via Ollama.

You can either reuse this index or rebuild it.

### 5.1 Rebuild the index (optional but recommended once)

Make sure Ollama is running and the embedding model is available, then:

```bash
python -m pythia.build_index
```

What this does:

1. Loads documents from `data/raw/wikipedia/`.
2. Chunks each document into overlapping character-based pieces.
3. Calls `ollama.embed` with `EMBEDDING_MODEL` to embed each chunk.
4. Writes a list of entries to `data/processed/pythia_index.json`.

This step demonstrates reproducible data processing and ensures the index matches the current settings.

---

## 6. Using pythia from the CLI

The simplest way to try the pipeline is the CLI:

```bash
python -m pythia.cli
```

Example:

```text
Loaded index with 780 entries from .../data/processed/pythia_index.json
pythia - simple RAG CLI. Ctrl+C to quit.

Question: What is data science?

Answer:
[model-generated answer based on retrieved context]

---- Retrieved context ----
[0.85] Data_science - Data science is an interdisciplinary field that uses scientific methods...
[0.79] Machine_learning - Machine learning is a field of inquiry that gives computers the ability...
...
```

Under the hood:

- `RAGPipeline` loads the index.
- BM25 preselects a set of relevant chunks.
- Embeddings via Ollama re-rank those chunks by cosine similarity.
- `tinyllama` generates an answer using only the retrieved context.

---

## 7. Running the FastAPI backend

To serve pythia behind a REST API:

```bash
uvicorn pythia.api:app --reload
```

This will:

- Load the index (`pythia_index.json`) once at startup.
- Construct a `RAGPipeline` instance.
- Serve the API on `http://127.0.0.1:8000`.

### Key endpoints

- **Health check**

  ```text
  GET /health
  ```

  Response:

  ```json
  {"status": "ok"}
  ```

- **Query**

  ```text
  POST /query
  ```

  Example request body:

  ```json
  {
    "question": "What is MLOps?",
    "top_n": 5
  }
  ```

  Example response shape:

  ```json
  {
    "question": "What is MLOps?",
    "answer": "MLOps is a set of practices for deploying and maintaining machine learning models in production...",
    "contexts": [
      {
        "doc_id": "MLOps",
        "title": "MLOps",
        "score": 0.87,
        "text": "..."
      },
      {
        "doc_id": "Machine_learning",
        "title": "Machine_learning",
        "score": 0.81,
        "text": "..."
      }
    ]
  }
  ```

You can explore and test the API interactively at:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Running the Streamlit UI

The Streamlit UI is a small web app that calls the FastAPI backend.

### 8.1 Start the API

In one terminal:

```bash
.venv\Scripts\activate
uvicorn pythia.api:app --reload
```

Make sure it’s running on `http://127.0.0.1:8000`.

### 8.2 Start the UI

In a second terminal:

```bash
.venv\Scripts\activate
streamlit run pythia/ui_streamlit.py
```

By default, `ui_streamlit.py` sends requests to:

```python
API_URL = "http://localhost:8000/query"
```

When Streamlit starts, it prints a URL similar to:

```text
Local URL: http://localhost:8501
```

Open that in your browser. You’ll see:

- A text input for your question,
- A slider for the number of context chunks (`top_n`),
- An **Ask** button,
- The model answer and the retrieved chunks rendered below.

You can change the question and number of chunks interactively to probe the behaviour of the retriever and generator.

---

## 9. Running containerised (API only, Ollama on host)

To satisfy the “containerised and behind an API” requirement, the `Dockerfile` builds a container for the **pythia API only**. Ollama still runs on the host and is accessed over HTTP.

This keeps the container lighter and avoids embedding the entire Ollama runtime.

### 9.1 Build the Docker image

From the project root:

```bash
docker build -t pythia-api .
```

The image:

- Uses `python:3.12-slim`,
- Installs dependencies from `requirements.txt`,
- Copies the code and data directory (including `data/processed/pythia_index.json`),
- Runs:

  ```bash
  uvicorn pythia.api:app --host 0.0.0.0 --port 8000
  ```

### 9.2 Ensure Ollama is running on the host in accordance with points 2. and 3.

### 9.3 Run the API container

On Docker Desktop, the special hostname `host.docker.internal` lets containers reach the host. We use it to let the API container talk to the host’s Ollama instance:

```bash
docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434  pythia-api
```

Now:

- The pythia API is available at `http://localhost:8000`.
- Inside the container, the `ollama` Python client connects to `OLLAMA_HOST`, i.e. the host’s Ollama server.

You can:

- Hit `http://localhost:8000/health` and `http://localhost:8000/docs`,
- Or run the local Streamlit UI and have it talk to the containerised API (no code changes needed: the UI only cares that `http://localhost:8000/query` is available).

---

## 10. Evaluation

A small evaluation set lives in `data/eval/eval_set.json`. Each entry has:

- an `id`,
- a `question`,
- and a reference `answer`.

To run the evaluation:

```bash
python -m pythia.evaluation
```

The script:

1. Loads the evaluation set.
2. Uses `RAGPipeline.answer()` for each question.
3. Computes:
   - token-level F1 between prediction and ground truth,
   - exact-match (EM) score.
4. Prints per-question scores and aggregate averages.

This is intentionally simple, but it demonstrates a reproducible evaluation loop and makes it easy to compare pipeline variants (e.g. different retriever settings).

---

## 11. Possible extensions

Some natural ways to make this solution accessible to end users (beyond the current minimal UI):

1. **Chat-style web UI**  
   Replace the simple Streamlit input with a chat interface that keeps history and shows which documents supported each answer.

2. **Chatbot integration**  
   Wrap the `/query` endpoint with a Slack or Microsoft Teams bot so users can “chat with the documentation” directly in their daily tools.

3. **Domain-specific views**  
   Serve separate endpoints / UIs per topic (e.g. “Risk & ERM”, “Data & AI”, “Energy & Telecom”) that only index corresponding documents and provide tailored examples.

These are not implemented here, but the existing API and modular RAG pipeline are designed so they could be added without major refactoring.
