#!/bin/bash
# Pre-commit hook: RAG-powered code review using local Gemma 4 26B
# Install: cp scripts/pre-commit-review.sh .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit

set -e

DIFF=$(git diff --cached --unified=5)
if [ -z "$DIFF" ]; then
  exit 0
fi

# Truncate massive diffs
DIFF_LINES=$(echo "$DIFF" | wc -l)
if [ "$DIFF_LINES" -gt 500 ]; then
  DIFF=$(echo "$DIFF" | head -500)
  DIFF="$DIFF\n\n... (truncated, ${DIFF_LINES} total lines changed)"
fi

echo "Running RAG code review on staged changes..."

REVIEW=$(python -c "
import sys, json, ollama, chromadb

diff = sys.stdin.read()

# Extract changed file paths for targeted retrieval
import re
files = re.findall(r'^\+\+\+ b/(.+)$', diff, re.MULTILINE)

# Retrieve relevant context from RAG
client = chromadb.PersistentClient(path=r'C:\Users\Matthew\gemma4-finetune\chroma_db')
col = client.get_collection('codebase')

context_chunks = []
for f in files[:5]:
    try:
        q = ollama.embed(model='nomic-embed-text', input=[f]).embeddings
        results = col.query(query_embeddings=q, n_results=3)
        context_chunks.extend(results['documents'][0])
    except:
        pass

context = '\n---\n'.join(context_chunks[:8])[:8000]

messages = [
    {'role': 'system', 'content': '''You are a code reviewer for the SRFM Trading Lab codebase.
Review the git diff against the existing codebase context.
Flag ONLY real issues:
- Breaking changes to existing APIs or interfaces
- Convention violations (naming, patterns, structure)
- Architectural inconsistencies with the existing codebase
- Missing error handling in critical paths
- Security issues (hardcoded secrets, injection vectors)
Do NOT flag: style preferences, minor formatting, or nitpicks.
If the diff looks clean, say \"LGTM\" and nothing else.
Keep your review under 10 lines.'''},
    {'role': 'user', 'content': f'Existing codebase context:\n{context}\n\n---\nDiff to review:\n{diff}'}
]

response = ollama.chat(model='gemma4-opt', messages=messages, options={'num_ctx': 8192})
print(response.message.content)
" <<< "$DIFF" 2>/dev/null)

if [ $? -ne 0 ]; then
  # Don't block commits if review fails
  echo "  (review unavailable, proceeding)"
  exit 0
fi

echo ""
echo "$REVIEW"
echo ""

# Only block on explicit BLOCK signals
if echo "$REVIEW" | grep -qi "BLOCK\|CRITICAL\|DO NOT COMMIT"; then
  echo "Review flagged critical issues. Commit blocked."
  echo "Override with: git commit --no-verify"
  exit 1
fi

exit 0
