# Local RAG Setup Guide: Codebase-Aware AI on Your Machine

Turn your local LLM into an AI that knows your entire codebase. No cloud APIs, no data leaving your machine, no token costs for retrieval. Works with any codebase in any language.

By the end of this guide you will have:
- A local 26B parameter LLM running on your GPU
- Your entire codebase chunked, embedded, and indexed in a vector database
- An HTTP API that retrieves relevant code and generates context-aware answers
- A git pre-commit hook that reviews every commit against your architecture
- Claude Code MCP integration so every AI session has your codebase as searchable memory
- VS Code editor integration for inline AI assistance

**Time to complete: ~1 hour**
**Hardware required: NVIDIA GPU with 8GB+ VRAM, 16GB+ RAM, 50GB+ free disk**

---

## Prerequisites

- Windows 10/11 (or Linux/macOS with minor path adjustments)
- NVIDIA GPU with CUDA support (RTX 3060+ recommended, we used RTX 4070 12GB)
- Python 3.10+
- Git
- ~50GB free disk space (model weights + vector database)

---

## Step 1: Install Ollama

Ollama runs local LLMs with one command. It handles model downloading, GPU memory management, and serves an OpenAI-compatible API.

1. Download from https://ollama.com/download
2. Run the installer (requires admin on Windows)
3. Verify it is running:

```bash
curl http://localhost:11434/api/tags
```

You should see `{"models":[]}` (empty model list).

## Step 2: Pull Your Models

You need two models: a large chat model for generation and a small embedding model for search.

```bash
# Chat model: Gemma 4 26B MoE (16.8GB download, needs 12GB+ VRAM)
# If you have less VRAM, use gemma3:12b (8GB VRAM) or gemma3:4b (4GB VRAM)
ollama pull gemma4:26b

# Embedding model: nomic-embed-text (274MB, runs on CPU)
ollama pull nomic-embed-text
```

Create an optimized profile with good defaults:

```bash
ollama create gemma4-opt -f - <<EOF
FROM gemma4:26b
PARAMETER num_ctx 8192
PARAMETER temperature 0.7
PARAMETER num_gpu 99
EOF
```

Verify both models are available:

```bash
ollama list
```

## Step 3: Install Python Dependencies

```bash
pip install chromadb ollama
```

That is it. Two packages. ChromaDB is the vector database. The ollama package is the Python client.

## Step 4: Index Your Codebase

This is the core step. You walk your source tree, split files into chunks at logical boundaries, embed each chunk, and store everything in ChromaDB.

Create a file called `rag_index.py`:

```python
"""
Index your codebase into ChromaDB for RAG retrieval.
Customize PROJECTS and EXTENSIONS for your setup.
"""

import os
import re
import time
import hashlib
from pathlib import Path
import chromadb
import ollama

# ---- CUSTOMIZE THESE ----

# Map of project_name -> absolute path to project root
PROJECTS = {
    "my-project": r"C:\path\to\your\project",
    # Add more projects here
}

# File extensions to index
EXTENSIONS = {
    ".rs", ".py", ".cpp", ".hpp", ".c", ".h", ".ts", ".tsx",
    ".go", ".jl", ".zig", ".ex", ".sql", ".md", ".toml", ".yaml",
}

# Directories to skip
SKIP_DIRS = {
    "target", "node_modules", "__pycache__", ".git", "venv",
    "build", "dist", ".next", "vendor",
}

# ---- CONFIG ----

CHUNK_SIZE = 1500       # characters per chunk (~375 tokens)
CHUNK_OVERLAP = 200     # overlap between adjacent chunks
DB_PATH = r"C:\Users\YOU\rag\chroma_db"  # where to store the database
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 50


def chunk_code(content, filepath, project, language):
    """Split code at function/class boundaries, then by size."""

    # Language-specific split patterns
    patterns = {
        "rust":       r"(?=^(?:pub\s+)?(?:fn |impl |struct |enum |mod |trait ))",
        "python":     r"(?=^(?:class |def |async def ))",
        "go":         r"(?=^(?:func |type ))",
        "cpp":        r"(?=^(?:class |struct |namespace |void |int |auto ))",
        "julia":      r"(?=^(?:function |struct |module ))",
    }

    pattern = patterns.get(language)
    if pattern:
        blocks = re.split(pattern, content, flags=re.MULTILINE)
        blocks = [b for b in blocks if b.strip()]
    else:
        blocks = [content]

    # Merge small blocks, split large ones
    chunks = []
    buffer = ""
    for block in blocks:
        if len(buffer) + len(block) < CHUNK_SIZE:
            buffer = (buffer + "\n\n" + block).strip() if buffer else block
        else:
            if buffer:
                chunks.append(buffer)
            if len(block) > CHUNK_SIZE:
                for i in range(0, len(block), CHUNK_SIZE - CHUNK_OVERLAP):
                    piece = block[i:i + CHUNK_SIZE]
                    if len(piece.strip()) > 50:
                        chunks.append(piece)
            else:
                buffer = block
                continue
            buffer = ""
    if buffer and len(buffer.strip()) > 50:
        chunks.append(buffer)

    # Add file path header to each chunk for context
    results = []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{filepath}:{i}:{chunk[:100]}".encode()).hexdigest()
        results.append({
            "id": chunk_id,
            "text": f"# {project}/{filepath}\n\n{chunk.strip()}",
            "metadata": {
                "project": project,
                "filepath": filepath,
                "language": language,
                "chunk_index": i,
            }
        })
    return results


def main():
    print("Collecting and chunking files...")

    all_chunks = []
    lang_map = {
        ".rs": "rust", ".py": "python", ".go": "go", ".cpp": "cpp",
        ".hpp": "cpp", ".c": "c", ".h": "c", ".ts": "typescript",
        ".tsx": "typescript", ".jl": "julia", ".zig": "zig",
        ".ex": "elixir", ".md": "markdown", ".toml": "toml",
        ".yaml": "yaml", ".yml": "yaml", ".sql": "sql",
    }

    for project_name, root_path in PROJECTS.items():
        if not os.path.exists(root_path):
            print(f"  SKIP {project_name}: not found")
            continue

        count = 0
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                           and not d.startswith(".")]

            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in EXTENSIONS:
                    continue

                filepath = os.path.join(dirpath, fname)
                try:
                    if os.path.getsize(filepath) > 500_000 or os.path.getsize(filepath) < 20:
                        continue
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except:
                    continue

                if len(content.strip()) < 50:
                    continue

                rel_path = os.path.relpath(filepath, root_path).replace("\\", "/")
                language = lang_map.get(ext, "text")
                chunks = chunk_code(content, rel_path, project_name, language)
                all_chunks.extend(chunks)
                count += 1

        print(f"  {project_name}: {count} files")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Pull embedding model
    print(f"Pulling {EMBED_MODEL}...")
    try:
        ollama.pull(EMBED_MODEL)
    except:
        pass

    # Store in ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("codebase")
    except:
        pass

    collection = client.create_collection(
        name="codebase",
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding {len(all_chunks)} chunks...")
    start = time.time()

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        try:
            response = ollama.embed(model=EMBED_MODEL, input=texts)
            collection.add(
                ids=ids,
                embeddings=response.embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"  Error on batch {i}: {e}")
            continue

        done = min(i + BATCH_SIZE, len(all_chunks))
        elapsed = time.time() - start
        rate = done / max(elapsed, 1)
        eta = (len(all_chunks) - done) / max(rate, 0.1)
        print(f"  {done}/{len(all_chunks)} ({rate:.0f}/s, ETA {eta:.0f}s)", end="\r")

    print(f"\nDone. {collection.count()} chunks indexed in {time.time()-start:.0f}s")
    print(f"Database: {DB_PATH}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python rag_index.py
```

This takes 1-5 minutes depending on codebase size. A 1M line codebase produces ~70K chunks and takes about 2 minutes.

## Step 5: Start the RAG API Server

This server sits between your tools and the LLM. Everything downstream talks to this.

Create `rag_server.py`:

```python
"""
RAG API Server. All downstream tools hit this.
"""

import json
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import chromadb
import ollama

DB_PATH = r"C:\Users\YOU\rag\chroma_db"  # match step 4
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4-opt"               # or whatever model you pulled
PORT = 11435
TOP_K = 8

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("codebase")


def retrieve(query, top_k=TOP_K, project=None):
    q_embed = ollama.embed(model=EMBED_MODEL, input=[query]).embeddings
    where = {"project": project} if project else None
    results = collection.query(query_embeddings=q_embed, n_results=top_k, where=where)
    return [{"text": doc, "project": meta["project"], "filepath": meta["filepath"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])]


def rag_query(query, system_prompt=None, project=None):
    chunks = retrieve(query, project=project)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)[:10000]
    system = system_prompt or "You are a senior engineer. Answer using the code context. Cite files."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\n---\nQuestion: {query}"}
    ]
    response = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192})
    sources = list(set(f"{c['project']}/{c['filepath']}" for c in chunks))
    return response.message.content, sources


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        path = self.path
        try:
            if path == "/v1/query":
                answer, sources = rag_query(body.get("query", ""), body.get("system"), body.get("project"))
                result = {"answer": answer, "sources": sources}

            elif path == "/v1/retrieve":
                chunks = retrieve(body.get("query", ""), body.get("top_k", TOP_K), body.get("project"))
                result = {"chunks": [{"filepath": f"{c['project']}/{c['filepath']}",
                                       "text": c["text"][:500]} for c in chunks]}

            elif path == "/v1/review":
                diff = body.get("diff", "")
                files = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
                chunks = []
                for f in files[:5]:
                    chunks.extend(retrieve(f, 3))
                context = "\n---\n".join(c["text"] for c in chunks)[:8000]
                messages = [
                    {"role": "system", "content": "Review this diff. Flag real issues only. Say LGTM if clean."},
                    {"role": "user", "content": f"Context:\n{context}\n\nDiff:\n{diff}"}
                ]
                resp = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192})
                result = {"review": resp.message.content, "files": files}

            elif path == "/v1/document":
                module = body.get("module_path", "")
                chunks = retrieve(module, 12)
                context = "\n---\n".join(c["text"] for c in chunks)[:10000]
                doc_type = body.get("type", "architecture")
                prompts = {
                    "architecture": "Generate architecture documentation for this module.",
                    "api": "Generate API reference documentation.",
                    "onboarding": "Generate an onboarding guide for a new developer.",
                }
                messages = [
                    {"role": "system", "content": "Generate documentation grounded in actual code. Only describe what exists."},
                    {"role": "user", "content": f"{prompts.get(doc_type, prompts['architecture'])}\n\nContext:\n{context}"}
                ]
                resp = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192})
                result = {"documentation": resp.message.content, "type": doc_type}

            elif path == "/v1/ideate":
                answer, sources = rag_query(
                    body.get("prompt", ""),
                    "Propose concrete ideas grounded in the existing architecture. Cite specific modules.",
                )
                result = {"ideas": answer, "sources": sources}

            else:
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "chunks": collection.count()}).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress access logs


if __name__ == "__main__":
    print(f"RAG API: http://localhost:{PORT}")
    print(f"  {collection.count()} chunks indexed")
    print(f"  Endpoints: /v1/query, /v1/retrieve, /v1/review, /v1/document, /v1/ideate")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
```

Start it:

```bash
python rag_server.py
```

Test it:

```bash
curl http://localhost:11435/health
# {"status": "ok", "chunks": 77154}

curl -X POST http://localhost:11435/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the main entry point work?"}'
```

## Step 6: Git Pre-Commit Review Hook

Every `git commit` now gets reviewed by the LLM against your codebase conventions.

Create `.git/hooks/pre-commit` in your repo:

```bash
#!/bin/bash
# RAG-powered code review on every commit
DIFF=$(git diff --cached --unified=5)
[ -z "$DIFF" ] && exit 0

# Truncate large diffs
[ $(echo "$DIFF" | wc -l) -gt 500 ] && DIFF=$(echo "$DIFF" | head -500)

REVIEW=$(curl -s -X POST http://localhost:11435/v1/review \
  -H "Content-Type: application/json" \
  -d "$(echo "$DIFF" | python -c 'import sys,json; print(json.dumps({"diff": sys.stdin.read()}))')" \
  2>/dev/null | python -c 'import sys,json; print(json.load(sys.stdin).get("review","(review unavailable)"))' 2>/dev/null)

[ $? -ne 0 ] && exit 0  # don't block if server is down

echo ""
echo "$REVIEW"
echo ""

echo "$REVIEW" | grep -qi "BLOCK\|CRITICAL" && echo "Blocked. Override: git commit --no-verify" && exit 1
exit 0
```

Make it executable:

```bash
chmod +x .git/hooks/pre-commit
```

Now every commit gets an architecture-aware review. Override with `git commit --no-verify` if needed.

## Step 7: Claude Code MCP Integration

This makes your codebase searchable from within Claude Code sessions.

Create `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "codebase-rag": {
      "command": "python",
      "args": ["C:\\path\\to\\your\\rag_mcp_server.py"],
      "env": {}
    }
  }
}
```

Create `rag_mcp_server.py`:

```python
"""
MCP Server: Exposes codebase RAG as tools for Claude Code.
"""

import json
import sys
import chromadb
import ollama

DB_PATH = r"C:\Users\YOU\rag\chroma_db"  # match step 4
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4-opt"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("codebase")


def retrieve(query, top_k=6):
    q_embed = ollama.embed(model=EMBED_MODEL, input=[query]).embeddings
    results = collection.query(query_embeddings=q_embed, n_results=top_k)
    return [{"text": doc, "project": meta["project"], "filepath": meta["filepath"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])]


TOOLS = [
    {
        "name": "codebase_search",
        "description": "Search the local codebase for code related to a concept, function, or pattern. Returns relevant code chunks with file paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "codebase_explain",
        "description": "Explain a module or function in the context of the full codebase.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code_or_path": {"type": "string", "description": "Code or path to explain"},
            },
            "required": ["code_or_path"],
        },
    },
]


def handle_tool(name, args):
    if name == "codebase_search":
        chunks = retrieve(args["query"])
        return "\n\n".join(
            f"## {c['project']}/{c['filepath']}\n```\n{c['text'][:800]}\n```"
            for c in chunks
        )
    elif name == "codebase_explain":
        chunks = retrieve(args["code_or_path"], 8)
        context = "\n---\n".join(c["text"] for c in chunks)[:10000]
        messages = [
            {"role": "system", "content": "Explain this in the context of the broader codebase."},
            {"role": "user", "content": f"Context:\n{context}\n\nExplain: {args['code_or_path']}"}
        ]
        return ollama.chat(model=CHAT_MODEL, messages=messages,
                           options={"num_ctx": 8192}).message.content
    return "Unknown tool"


def send(msg):
    data = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n{data}")
    sys.stdout.flush()


def read():
    headers = {}
    while True:
        line = sys.stdin.readline()
        if line in ("\r\n", "\n"):
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    length = int(headers.get("Content-Length", 0))
    return json.loads(sys.stdin.read(length)) if length else {}


def main():
    while True:
        try:
            msg = read()
        except:
            break

        method = msg.get("method", "")
        mid = msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "codebase-rag", "version": "1.0.0"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            result = handle_tool(msg["params"]["name"], msg["params"].get("arguments", {}))
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": result}]
            }})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})


if __name__ == "__main__":
    main()
```

Restart Claude Code. It will now have `codebase_search` and `codebase_explain` as available tools. When you ask questions about your code, Claude will call these tools automatically to retrieve relevant context from your local index.

## Step 8: VS Code Integration (Optional)

Install the Continue.dev extension in VS Code, then add to your `.vscode/settings.json`:

```json
{
  "continue.models": [
    {
      "title": "Local Gemma (RAG)",
      "provider": "ollama",
      "model": "gemma4-opt",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

Now you can highlight code in VS Code, press Ctrl+L, and ask questions answered by your local LLM.

---

## Step 9: Re-index When Your Code Changes

The vector index is a snapshot. When you make significant changes, re-run the indexer:

```bash
python rag_index.py
```

Takes 1-5 minutes. You could automate this as a post-merge hook or a daily cron job:

```bash
# .git/hooks/post-merge
#!/bin/bash
echo "Re-indexing codebase for RAG..."
python /path/to/rag_index.py &
```

---

## Architecture Summary

```
Your Code
  |
  v
rag_index.py -----> ChromaDB (vector database, on disk)
                        |
                        v
                   rag_server.py (HTTP :11435)
                   /    |    \       \
                  /     |     \       \
          Claude Code  git    VS Code  curl/scripts
          (MCP tools)  hook   Continue  (any HTTP client)
                  \     |     /       /
                   \    |    /       /
                    v   v   v       v
                   Ollama (LLM :11434)
                   Gemma 4 26B on GPU
```

## What Each Piece Does

| Component | What it does | Port/Path |
|---|---|---|
| Ollama | Runs the LLM on your GPU | :11434 |
| ChromaDB | Stores code embeddings on disk | ~/rag/chroma_db |
| rag_index.py | Chunks and embeds your code | run manually |
| rag_server.py | HTTP API for all downstream tools | :11435 |
| rag_mcp_server.py | Exposes tools to Claude Code | stdio (MCP) |
| pre-commit hook | Reviews every commit | .git/hooks/ |
| Continue.dev | VS Code inline AI | extension |

## Troubleshooting

**"Ollama connection refused"**
Ollama is not running. Start the Ollama desktop app or run `ollama serve`.

**"Collection not found"**
You have not run the indexer yet. Run `python rag_index.py`.

**Slow responses (>60s)**
Your GPU is overloaded. Check `nvidia-smi`. If Ollama and another GPU process are competing, close one. Alternatively, use a smaller model (`gemma3:4b` responds in 5-10s).

**Pre-commit hook blocks commit**
Override with `git commit --no-verify`. If it happens often, adjust the system prompt in the review endpoint to be less strict.

**MCP tools not appearing in Claude Code**
Restart Claude Code after creating `~/.claude/mcp.json`. Check that the python path in the config is correct and that `chromadb` and `ollama` are installed.

**Stale results after code changes**
Re-run `python rag_index.py`. The old index is deleted and rebuilt from scratch.

## Cost

Zero. Everything runs locally.

| Resource | Cost |
|---|---|
| Ollama | Free, open source |
| Gemma 4 26B | Free, Apache 2.0 license |
| ChromaDB | Free, open source |
| nomic-embed-text | Free, Apache 2.0 license |
| GPU electricity | ~200W while querying, idle otherwise |
| Cloud API tokens | $0 |
