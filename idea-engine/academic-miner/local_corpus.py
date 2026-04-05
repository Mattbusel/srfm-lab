"""
local_corpus.py — Local Research Paper Corpus Miner
====================================================
Scans a local directory of PDFs and plain-text files, extracts the text,
and runs IdeaExtractor on each document.  Results are cached in the
``academic_papers`` table so documents are not re-processed on subsequent
runs.

PDF extraction
--------------
Uses ``pdfminer.six`` when available (``pip install pdfminer.six``).
Falls back gracefully to treating the file as plain text (or skipping) if
pdfminer is not installed.

Usage
-----
    miner = LocalCorpusMiner(db_path="idea_engine.db")
    results = miner.run()
    for doc in results:
        print(doc.title, "—", doc.n_candidates, "ideas extracted")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .idea_extractor import IdeaCandidate, IdeaExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CORPUS_DIR: str = r"C:\Users\Matthew\srfm-lab\research\papers"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".txt", ".md", ".tex")

# Max characters to pass to IdeaExtractor (abstracts tend to be first 2 KB)
MAX_TEXT_CHARS: int = 8_000

# Cache sentinel stored in DB for already-processed files
PROCESSED_SENTINEL: str = "local_processed"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CorpusDocument:
    """Represents a single processed local document."""

    file_path:    str
    title:        str
    abstract:     str
    full_text:    str                = field(repr=False, default="")
    file_hash:    str                = ""           # SHA-256 of file bytes
    n_candidates: int                = 0
    candidates:   List[IdeaCandidate] = field(default_factory=list, repr=False)
    db_id:        Optional[int]      = None

    def to_dict(self) -> dict:
        return {
            "file_path":    self.file_path,
            "title":        self.title,
            "abstract":     self.abstract[:500],
            "file_hash":    self.file_hash,
            "n_candidates": self.n_candidates,
        }

    def __repr__(self) -> str:
        return (
            f"CorpusDocument(title={self.title[:50]!r}, "
            f"ideas={self.n_candidates}, hash={self.file_hash[:8]})"
        )


# ---------------------------------------------------------------------------
# LocalCorpusMiner
# ---------------------------------------------------------------------------

class LocalCorpusMiner:
    """
    Scans a local directory for research documents and extracts ideas.

    Parameters
    ----------
    corpus_dir : str
        Directory to scan for PDFs and text files.
    db_path : str
        SQLite database path.
    force_reprocess : bool
        If True, reprocess files even if already cached.
    """

    def __init__(
        self,
        corpus_dir: str = DEFAULT_CORPUS_DIR,
        db_path: str = "idea_engine.db",
        force_reprocess: bool = False,
    ) -> None:
        self.corpus_dir      = Path(corpus_dir)
        self.db_path         = db_path
        self.force_reprocess = force_reprocess
        self._db: Optional[sqlite3.Connection] = None
        self._extractor      = IdeaExtractor(db_path=db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.db_path)
            self._db.row_factory = sqlite3.Row
        return self._db

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS academic_papers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT    NOT NULL,
                paper_id        TEXT    UNIQUE,
                title           TEXT    NOT NULL,
                authors         TEXT,
                abstract        TEXT,
                relevance_score REAL,
                categories      TEXT,
                url             TEXT,
                mined_at        TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
            CREATE TABLE IF NOT EXISTS hypothesis_candidates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id INTEGER REFERENCES academic_papers(id),
                hypothesis_text TEXT    NOT NULL,
                mapped_component TEXT,
                param_suggestions TEXT,
                confidence      REAL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                created_at      TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );
        """)
        conn.commit()

    def _is_cached(self, file_hash: str) -> Optional[int]:
        """
        Return the DB row id if the file has already been processed,
        else None.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT id FROM academic_papers WHERE source='local' AND paper_id=?",
            (file_hash,),
        ).fetchone()
        return row["id"] if row else None

    def _store_document(self, doc: CorpusDocument) -> int:
        """Insert a CorpusDocument into academic_papers and return its id."""
        conn = self._connect()
        now  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur  = conn.execute(
            """
            INSERT OR IGNORE INTO academic_papers
                (source, paper_id, title, authors, abstract,
                 relevance_score, categories, url, mined_at)
            VALUES ('local', ?, ?, NULL, ?, NULL, NULL, ?, ?)
            """,
            (doc.file_hash, doc.title, doc.abstract[:2000], doc.file_path, now),
        )
        conn.commit()
        if cur.lastrowid:
            doc.db_id = cur.lastrowid
        else:
            row = conn.execute(
                "SELECT id FROM academic_papers WHERE paper_id=?", (doc.file_hash,)
            ).fetchone()
            if row:
                doc.db_id = row["id"]
        return doc.db_id or 0

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def iter_files(self) -> Iterator[Path]:
        """
        Yield all supported document files under self.corpus_dir.

        Recursively walks subdirectories.

        Yields
        ------
        Path
        """
        if not self.corpus_dir.exists():
            logger.warning("Corpus directory does not exist: %s", self.corpus_dir)
            return

        for root, dirs, files in os.walk(self.corpus_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in sorted(files):
                path = Path(root) / fname
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield path

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute SHA-256 of file bytes (first 512 KB for large files)."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                chunk = f.read(512 * 1024)
                h.update(chunk)
        except OSError:
            h.update(str(path).encode())
        return h.hexdigest()

    def extract_text(self, path: Path) -> str:
        """
        Extract plain text from a file.

        - .pdf  → pdfminer.six if available, otherwise reads raw bytes
                  and decodes printable ASCII (lossy fallback).
        - .txt / .md / .tex → read as UTF-8 with error replacement.

        Parameters
        ----------
        path : Path

        Returns
        -------
        str
            Extracted text, limited to MAX_TEXT_CHARS characters.
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        else:
            return self._extract_text_file(path)

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from a PDF using pdfminer.six or raw ASCII fallback."""
        # --- Try pdfminer.six ---
        try:
            from pdfminer.high_level import extract_text as pm_extract  # type: ignore
            text = pm_extract(str(path))
            text = re.sub(r"\s+", " ", text or "").strip()
            logger.debug("pdfminer extracted %d chars from %s", len(text), path.name)
            return text[:MAX_TEXT_CHARS]
        except ImportError:
            logger.debug("pdfminer.six not available; using ASCII fallback for %s", path.name)
        except Exception as exc:
            logger.warning("pdfminer failed for %s: %s", path.name, exc)

        # --- Raw ASCII fallback ---
        try:
            with open(path, "rb") as f:
                raw = f.read(128 * 1024)
            # Keep only printable ASCII bytes
            text = "".join(chr(b) if 32 <= b < 127 or b in (9, 10, 13) else " " for b in raw)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:MAX_TEXT_CHARS]
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return ""

    @staticmethod
    def _extract_text_file(path: Path) -> str:
        """Read a plain text file and return its content."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_TEXT_CHARS)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return ""

    # ------------------------------------------------------------------
    # Title / abstract heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_title(text: str, filename: str) -> str:
        """
        Heuristically extract a title from document text.

        Strategy:
        1. Look for lines before "Abstract" that look like a title
           (capitalised, < 200 chars, not all-caps boilerplate).
        2. Fall back to the filename stem.
        """
        # Try first non-empty line that's reasonably title-like
        for line in text.split("\n")[:30]:
            line = line.strip()
            if 10 < len(line) < 200:
                if not re.match(r"^(Abstract|Introduction|Keywords|Author)", line, re.I):
                    # Heuristic: title-like if mostly capitalised words
                    words = line.split()
                    if words and sum(1 for w in words if w and w[0].isupper()) / len(words) > 0.5:
                        return line
        # Fall back to filename
        return Path(filename).stem.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _guess_abstract(text: str) -> str:
        """
        Extract the abstract section from document text.

        Looks for the "Abstract" heading and captures the following
        paragraph.
        """
        # Pattern: "Abstract\n..." or "Abstract — ..."
        m = re.search(
            r"\bAbstract\b[:\s—–-]*\n?\s*([A-Z][^§]{100,2000}?)(?:\n\s*\n|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()[:2000]

        # Fallback: return the first 800 chars of clean body text
        clean = re.sub(r"\s+", " ", text[:3000]).strip()
        return clean[:800]

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_file(self, path: Path) -> Optional[CorpusDocument]:
        """
        Process a single file: hash → cache check → extract → mine ideas.

        Parameters
        ----------
        path : Path

        Returns
        -------
        CorpusDocument or None
        """
        fhash = self._file_hash(path)

        if not self.force_reprocess:
            existing_id = self._is_cached(fhash)
            if existing_id is not None:
                logger.debug("Cache hit for %s (db_id=%d)", path.name, existing_id)
                return None  # already processed

        logger.info("Processing %s …", path.name)
        text = self.extract_text(path)
        if not text or len(text) < 50:
            logger.warning("Insufficient text from %s — skipping.", path.name)
            return None

        title    = self._guess_title(text, str(path))
        abstract = self._guess_abstract(text)

        doc = CorpusDocument(
            file_path  = str(path),
            title      = title,
            abstract   = abstract,
            full_text  = text,
            file_hash  = fhash,
        )

        # Store in DB first to get a db_id
        db_id = self._store_document(doc)

        # Extract ideas
        candidates = self._extractor.extract_hypothesis(abstract)
        if candidates:
            self._extractor.store_candidates(candidates, source_paper_id=db_id)
        doc.candidates   = candidates
        doc.n_candidates = len(candidates)

        logger.info(
            "  %s → %d ideas extracted (db_id=%d)",
            path.name, len(candidates), db_id,
        )
        return doc

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> List[CorpusDocument]:
        """
        Process all files in the corpus directory.

        Returns
        -------
        List[CorpusDocument]
            Documents that were processed (not returned from cache).
        """
        if not self.corpus_dir.exists():
            logger.warning("Corpus dir not found: %s — creating placeholder.", self.corpus_dir)
            try:
                self.corpus_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            return []

        processed: List[CorpusDocument] = []
        for fpath in self.iter_files():
            doc = self.process_file(fpath)
            if doc:
                processed.append(doc)

        total_ideas = sum(d.n_candidates for d in processed)
        logger.info(
            "LocalCorpusMiner run complete: %d new docs, %d total ideas.",
            len(processed), total_ideas,
        )
        return processed

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_cached(self) -> List[dict]:
        """
        Return all locally-sourced papers stored in the DB.

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM academic_papers WHERE source='local' ORDER BY mined_at DESC"
        )
        return [dict(r) for r in cur]

    def search_corpus(self, keyword: str) -> List[dict]:
        """
        Full-text search the cached local corpus abstracts.

        Parameters
        ----------
        keyword : str
            Substring to search for in title or abstract.

        Returns
        -------
        List[dict]
        """
        conn = self._connect()
        pattern = f"%{keyword.lower()}%"
        cur = conn.execute(
            """
            SELECT * FROM academic_papers
            WHERE source='local'
              AND (lower(title) LIKE ? OR lower(abstract) LIKE ?)
            ORDER BY mined_at DESC
            """,
            (pattern, pattern),
        )
        return [dict(r) for r in cur]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._extractor.close()
        if self._db:
            self._db.close()
            self._db = None

    def __repr__(self) -> str:
        return f"LocalCorpusMiner(dir={self.corpus_dir!r}, db={self.db_path!r})"

    def __enter__(self) -> "LocalCorpusMiner":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS_DIR
    with LocalCorpusMiner(corpus_dir=corpus_dir) as miner:
        docs = miner.run()
        print(f"\nProcessed {len(docs)} new documents:")
        for d in docs:
            print(f"  {d.title[:60]:<60}  ideas={d.n_candidates}")
