"""
RAG API Server: HTTP endpoints for all downstream integrations.
Serves as the single integration point for editor, CI, Claude agents, doc gen, and ideation.

Endpoints:
  POST /v1/query         - General RAG query (retrieve + generate)
  POST /v1/retrieve      - Retrieve code chunks only (no LLM)
  POST /v1/review        - Code review (diff + context)
  POST /v1/explain       - Explain a code block
  POST /v1/test          - Generate tests for code
  POST /v1/document      - Generate documentation for a module
  POST /v1/ideate        - Strategy/architecture ideation
  POST /v1/references    - Find what calls/uses a function or type
  GET  /v1/stats         - Index statistics
  GET  /health           - Health check
"""

import json
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import chromadb
import ollama

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = r"C:\Users\Matthew\gemma4-finetune\chroma_db"
COLLECTION_NAME = "codebase"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4-opt"
PORT = 11435
TOP_K = 8
MAX_CONTEXT = 10000

# ── Init ──────────────────────────────────────────────────────────────────────

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)
print(f"Loaded {collection.count()} chunks from ChromaDB")


def retrieve(query: str, top_k: int = TOP_K, project: str = None) -> list[dict]:
    q_embed = ollama.embed(model=EMBED_MODEL, input=[query]).embeddings
    where = {"project": project} if project else None
    results = collection.query(query_embeddings=q_embed, n_results=top_k, where=where)
    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append({"text": doc, "project": meta["project"], "filepath": meta["filepath"],
                        "language": meta["language"], "distance": float(dist)})
    return chunks


def generate(system: str, user: str, context_chunks: list[dict] = None) -> str:
    context = ""
    if context_chunks:
        context = "\n\n---\n\n".join(c["text"] for c in context_chunks[:TOP_K])[:MAX_CONTEXT]
        user = f"Codebase context:\n{context}\n\n---\n{user}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    response = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192})
    return response.message.content


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_query(body: dict) -> dict:
    query = body.get("query", "")
    project = body.get("project")
    chunks = retrieve(query, body.get("top_k", TOP_K), project)
    answer = generate(
        "You are a senior engineer with deep knowledge of this codebase. Cite specific files and functions.",
        f"Question: {query}",
        chunks,
    )
    return {"answer": answer, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks]}


def handle_retrieve(body: dict) -> dict:
    query = body.get("query", "")
    chunks = retrieve(query, body.get("top_k", TOP_K), body.get("project"))
    return {"chunks": [{"filepath": f"{c['project']}/{c['filepath']}", "language": c["language"],
                         "distance": c["distance"], "text": c["text"][:500]} for c in chunks]}


def handle_review(body: dict) -> dict:
    diff = body.get("diff", "")
    files = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
    chunks = []
    for f in files[:5]:
        chunks.extend(retrieve(f, 3))

    answer = generate(
        """You are a code reviewer for the SRFM Trading Lab.
Flag only real issues: breaking API changes, convention violations, architectural inconsistencies, missing error handling, security issues.
If the diff is clean, say "LGTM". Keep it under 10 lines.""",
        f"Diff to review:\n{diff}",
        chunks,
    )
    return {"review": answer, "files_reviewed": files, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks[:5]]}


def handle_explain(body: dict) -> dict:
    code = body.get("code", "")
    filepath = body.get("filepath", "")
    chunks = retrieve(f"{filepath}\n{code[:200]}", TOP_K)
    answer = generate(
        "Explain this code in the context of the broader codebase. Reference related modules and how this fits into the architecture.",
        f"File: {filepath}\n\nCode:\n{code}",
        chunks,
    )
    return {"explanation": answer, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks]}


def handle_test(body: dict) -> dict:
    code = body.get("code", "")
    filepath = body.get("filepath", "")
    language = body.get("language", "python")
    chunks = retrieve(f"test {filepath}\n{code[:200]}", TOP_K)
    answer = generate(
        f"Generate comprehensive unit tests for this {language} code. Match the testing style used in the codebase (check retrieved test files for conventions). Include edge cases.",
        f"File: {filepath}\n\nCode to test:\n{code}",
        chunks,
    )
    return {"tests": answer, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks]}


def handle_document(body: dict) -> dict:
    module_path = body.get("module_path", "")
    doc_type = body.get("type", "architecture")  # architecture / api / onboarding
    chunks = retrieve(module_path, 12)

    prompts = {
        "architecture": "Generate an architecture document for this module. Cover: purpose, key components, data flow, dependencies, design decisions, and integration points.",
        "api": "Generate API reference documentation. List every public function/method/type with parameters, return types, and usage examples.",
        "onboarding": "Generate an onboarding guide for a new developer. Explain what this module does, how to work with it, key concepts to understand, and common tasks.",
    }
    prompt = prompts.get(doc_type, prompts["architecture"])

    answer = generate(
        "You are a technical writer generating documentation grounded in actual code. Only describe what exists in the code. Do not invent features.",
        f"{prompt}\n\nModule: {module_path}",
        chunks,
    )
    return {"documentation": answer, "type": doc_type, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks]}


def handle_ideate(body: dict) -> dict:
    prompt = body.get("prompt", "")
    constraints = body.get("constraints", "")
    chunks = retrieve(prompt, 10)
    answer = generate(
        """You are a quantitative research advisor with full knowledge of the SRFM Trading Lab codebase.
Propose concrete, implementable ideas grounded in the existing architecture.
For each idea: name it, explain the hypothesis, specify which existing modules it builds on, and outline the implementation steps.""",
        f"Ideation prompt: {prompt}\n\nConstraints: {constraints}" if constraints else f"Ideation prompt: {prompt}",
        chunks,
    )
    return {"ideas": answer, "sources": [f"{c['project']}/{c['filepath']}" for c in chunks]}


def handle_references(body: dict) -> dict:
    name = body.get("name", "")  # function or type name
    chunks = retrieve(name, 15)
    # Filter to chunks that actually contain the name
    matching = [c for c in chunks if name in c["text"]]
    return {
        "name": name,
        "found_in": [{"filepath": f"{c['project']}/{c['filepath']}", "language": c["language"]} for c in matching],
        "total_references": len(matching),
    }


def handle_stats(body: dict) -> dict:
    return {
        "total_chunks": collection.count(),
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "db_path": DB_PATH,
    }


# ── HTTP Server ───────────────────────────────────────────────────────────────

ROUTES = {
    "/v1/query": handle_query,
    "/v1/retrieve": handle_retrieve,
    "/v1/review": handle_review,
    "/v1/explain": handle_explain,
    "/v1/test": handle_test,
    "/v1/document": handle_document,
    "/v1/ideate": handle_ideate,
    "/v1/references": handle_references,
}


class RAGHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ROUTES:
            self.send_error(404, f"Unknown endpoint: {path}")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

        try:
            start = time.time()
            result = ROUTES[path](body)
            result["elapsed_ms"] = int((time.time() - start) * 1000)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "chunks": collection.count()}).encode("utf-8"))
        elif path == "/v1/stats":
            result = handle_stats({})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        print(f"  {self.address_string()} {args[0]}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"RAG API Server starting on http://localhost:{PORT}")
    print(f"  ChromaDB: {collection.count()} chunks")
    print(f"  Embed: {EMBED_MODEL}")
    print(f"  Chat: {CHAT_MODEL}")
    print(f"\nEndpoints:")
    print(f"  POST /v1/query      - General RAG query")
    print(f"  POST /v1/retrieve   - Code chunk retrieval")
    print(f"  POST /v1/review     - Code review (send diff)")
    print(f"  POST /v1/explain    - Explain code block")
    print(f"  POST /v1/test       - Generate tests")
    print(f"  POST /v1/document   - Generate docs")
    print(f"  POST /v1/ideate     - Strategy ideation")
    print(f"  POST /v1/references - Find references")
    print(f"  GET  /v1/stats      - Index stats")
    print(f"  GET  /health        - Health check")

    server = HTTPServer(("0.0.0.0", PORT), RAGHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
