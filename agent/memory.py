"""NIA's long-term memory: facts she's asked to remember and a log of past exchanges, kept in SQLite and
found by meaning (vector search), so each turn sends the LLM only the few memories that matter instead of
the whole history.

Embeddings come from EmbeddingGemma on the CPU through Ollama. It was chosen against all-minilm,
granite-embedding and nomic-embed-text because it alone separated relevant memories (similarity >= 0.29)
from off-topic queries (<= 0.16), so unrelated turns get no memories at all. If Ollama is down, memory
switches itself off for that call and NIA carries on without it.
"""
import json
import logging
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DB = Path(__file__).resolve().parent.parent / "memory" / "nia_memory.sqlite"
# EmbeddingGemma's own prompts for stored documents and for searches
DOC_PREFIX, QUERY_PREFIX = "title: none | text: ", "task: search result | query: "


class Memory:
    def __init__(self, model, path=DB, embed=None):
        """model: the Ollama embedding model. embed: replaces Ollama (tests)."""
        self.model = model
        self._embed = embed or self._ollama_embed
        self.down_until = 0.0  # after a failed embedding, memory waits this long before trying again
        path.parent.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, kind TEXT, text TEXT,"
                        " created REAL, vector BLOB)")
        rows = self.db.execute("SELECT id, kind, text, created, vector FROM memories").fetchall()
        self.rows = [(i, kind, text, created) for i, kind, text, created, _ in rows]
        # ponytail: brute-force cosine over every memory - milliseconds into the thousands; use sqlite-vec or
        # FAISS if this grows past ~100k
        self.vectors = np.array([np.frombuffer(v, dtype=np.float32) for *_, v in rows]) if rows else None

    def _ollama_embed(self, texts):
        body = {"model": self.model, "input": texts, "options": {"num_gpu": 0}}  # CPU: the GPU is Bonsai's
        request = urllib.request.Request("http://127.0.0.1:11434/api/embed", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        vectors = np.array(json.load(urllib.request.urlopen(request, timeout=30))["embeddings"], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _vector(self, text, query):
        if time.time() < self.down_until:  # each refused call costs ~2 s on Windows - don't pay it every turn
            return None
        try:
            return self._embed([(QUERY_PREFIX if query else DOC_PREFIX) + text])[0]
        except (OSError, KeyError, ValueError) as e:
            logger.warning(f"Embedding model unavailable ({e!r}) - memory off for a minute")
            self.down_until = time.time() + 60
            return None

    def add(self, text, kind):
        vector = self._vector(text, query=False)
        if vector is None:
            return None
        created = time.time()
        cursor = self.db.execute("INSERT INTO memories (kind, text, created, vector) VALUES (?, ?, ?, ?)",
                                 (kind, text, created, vector.astype(np.float32).tobytes()))
        self.db.commit()
        self.rows.append((cursor.lastrowid, kind, text, created))
        self.vectors = vector[None] if self.vectors is None else np.vstack([self.vectors, vector])
        return cursor.lastrowid

    def search(self, query, k=4, min_similarity=0.22, exclude=()):
        """[(similarity, id, kind, text, created)] most like the query, best first; nothing below min_similarity"""
        if self.vectors is None:
            return []
        vector = self._vector(query, query=True)
        if vector is None:
            return []
        sims = self.vectors @ vector
        hits = []
        for i in np.argsort(-sims):
            if sims[i] < min_similarity or len(hits) == k:
                break
            if self.rows[i][0] not in exclude:
                hits.append((float(sims[i]), *self.rows[i]))
        return hits

    def forget(self, about, min_similarity=0.3):
        """Delete the remembered facts that best match `about`; returns their text"""
        doomed = [hit for hit in self.search(about, k=3, min_similarity=min_similarity) if hit[2] == "fact"]
        for _, memory_id, *_ in doomed:
            self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            index = next(n for n, row in enumerate(self.rows) if row[0] == memory_id)
            del self.rows[index]
            self.vectors = np.delete(self.vectors, index, axis=0) if len(self.rows) else None
        self.db.commit()
        return [text for *_, text, _ in doomed]

    def tools(self):
        @tool
        def remember(fact: str) -> str:
            """Save a fact to remember long-term, e.g. "Aalok's sister Priya's birthday is June 3".
            Use when the user asks you to remember something, or tells you something worth keeping.
            Write it as a standalone sentence that makes sense on its own later"""
            return "Remembered." if self.add(fact, "fact") else "Couldn't save that - memory is unavailable."

        @tool
        def forget(about: str) -> str:
            """Delete remembered facts about something, when the user asks you to forget it"""
            gone = self.forget(about)
            return f"Forgot: {'; '.join(gone)}" if gone else "Nothing like that was remembered."

        return [remember, forget]


def note(hits):
    """The memories as a short block appended to the user's message - after the stable system prompt,
    so llama-server's prompt cache still works"""
    if not hits:
        return ""
    lines = [f"- ({datetime.fromtimestamp(created):%d %b %Y}) {text}" for _, _, _, text, created in hits]
    return "\n[From memory - use only if relevant, don't read out]\n" + "\n".join(lines)
