"""
MCP Server: Exposes codebase RAG as tools for Claude Code.
Claude agents can call these tools to query the local codebase
instead of stuffing everything into the context window.

Tools:
  - codebase_search: Search for code by concept/keyword
  - codebase_explain: Explain a module or function in context
  - codebase_references: Find all usages of a type/function
  - codebase_review: Review a code diff against conventions
  - codebase_ideate: Generate implementation ideas
"""

import json
import sys
import chromadb
import ollama

DB_PATH = r"C:\Users\Matthew\gemma4-finetune\chroma_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4-opt"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection("codebase")


def retrieve(query: str, top_k: int = 6, project: str = None) -> list[dict]:
    q_embed = ollama.embed(model=EMBED_MODEL, input=[query]).embeddings
    where = {"project": project} if project else None
    results = collection.query(query_embeddings=q_embed, n_results=top_k, where=where)
    return [{"text": doc, "project": meta["project"], "filepath": meta["filepath"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])]


def rag_generate(system: str, user: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)[:10000]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Codebase context:\n{context}\n\n---\n{user}"}
    ]
    return ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192}).message.content


# ── MCP Protocol ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "codebase_search",
        "description": "Search the user's private codebase (1.25M LOC across 15 projects) for code related to a concept, function, type, or pattern. Returns relevant code chunks with file paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for (concept, function name, pattern)"},
                "project": {"type": "string", "description": "Optional: filter to a specific project name"},
                "top_k": {"type": "integer", "description": "Number of results (default 6)", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "codebase_explain",
        "description": "Explain a module, function, or code block in the context of the full codebase architecture. Uses RAG to find related components.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code_or_path": {"type": "string", "description": "Code block or file path to explain"},
            },
            "required": ["code_or_path"],
        },
    },
    {
        "name": "codebase_references",
        "description": "Find all places in the codebase where a specific function, type, or variable is used or defined.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function, type, or variable name to find"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "codebase_review",
        "description": "Review a code diff against the existing codebase conventions and architecture.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "diff": {"type": "string", "description": "Git diff to review"},
            },
            "required": ["diff"],
        },
    },
    {
        "name": "codebase_ideate",
        "description": "Generate implementation ideas grounded in the existing codebase architecture.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to ideate about"},
            },
            "required": ["prompt"],
        },
    },
]


def handle_tool_call(name: str, arguments: dict) -> str:
    if name == "codebase_search":
        chunks = retrieve(arguments["query"], arguments.get("top_k", 6), arguments.get("project"))
        results = []
        for c in chunks:
            results.append(f"## {c['project']}/{c['filepath']}\n```\n{c['text'][:800]}\n```")
        return "\n\n".join(results)

    elif name == "codebase_explain":
        query = arguments["code_or_path"]
        chunks = retrieve(query, 8)
        return rag_generate(
            "Explain this code in the context of the broader codebase. Reference related modules.",
            f"Explain: {query}", chunks)

    elif name == "codebase_references":
        name_q = arguments["name"]
        chunks = retrieve(name_q, 15)
        matching = [c for c in chunks if name_q in c["text"]]
        if not matching:
            return f"No references found for '{name_q}'"
        lines = [f"- {c['project']}/{c['filepath']}" for c in matching]
        return f"Found {len(matching)} references to `{name_q}`:\n" + "\n".join(lines)

    elif name == "codebase_review":
        diff = arguments["diff"]
        import re
        files = re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE)
        chunks = []
        for f in files[:5]:
            chunks.extend(retrieve(f, 3))
        return rag_generate(
            "Review this diff against the codebase conventions. Flag real issues only. Say LGTM if clean.",
            f"Diff:\n{diff}", chunks)

    elif name == "codebase_ideate":
        chunks = retrieve(arguments["prompt"], 10)
        return rag_generate(
            "Propose concrete, implementable ideas grounded in the existing architecture.",
            f"Ideation: {arguments['prompt']}", chunks)

    return "Unknown tool"


# ── MCP stdio protocol ────────────────────────────────────────────────────────

def send_message(msg: dict):
    data = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n{data}")
    sys.stdout.flush()


def read_message() -> dict:
    headers = {}
    while True:
        line = sys.stdin.readline()
        if line == "\r\n" or line == "\n":
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
    length = int(headers.get("Content-Length", 0))
    if length > 0:
        data = sys.stdin.read(length)
        return json.loads(data)
    return {}


def main():
    while True:
        try:
            msg = read_message()
        except (EOFError, KeyboardInterrupt):
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "initialize":
            send_message({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "codebase-rag", "version": "1.0.0"},
                }
            })

        elif method == "notifications/initialized":
            pass  # no response needed

        elif method == "tools/list":
            send_message({
                "jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": TOOLS}
            })

        elif method == "tools/call":
            tool_name = msg["params"]["name"]
            arguments = msg["params"].get("arguments", {})
            try:
                result_text = handle_tool_call(tool_name, arguments)
                send_message({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": result_text}]}
                })
            except Exception as e:
                send_message({
                    "jsonrpc": "2.0", "id": msg_id,
                    "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
                })

        elif method == "ping":
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {}})


if __name__ == "__main__":
    main()
