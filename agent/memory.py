"""NIA's long-term memory: facts she's asked to remember, skills (how to do a multi-step job, saved by name to
repeat it) and a log of past exchanges, kept in SQLite and found by meaning (vector search), so each turn sends
the LLM only the few memories that matter instead of the whole history.

Embeddings come from EmbeddingGemma on the CPU through Ollama. It was chosen against all-minilm,
granite-embedding and nomic-embed-text because it alone separated relevant memories (similarity >= 0.29)
from off-topic queries (<= 0.16), so unrelated turns get no memories at all. If Ollama is down, memory
switches itself off for that call and NIA carries on without it.
"""
import json
import logging
import re
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DB = Path(__file__).resolve().parent.parent / "memory" / "nia_memory.sqlite"
DUPLICATE = 0.88  # memories at least this alike are the same thing said twice
# Past exchanges need a closer match than facts: the ones recalled at 0.24-0.28 were nearly all off-topic
# ("Sat down" brought back a volume change), and an old answer like "X is playing" reads as current
EXCHANGE_MIN = 0.30
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
        hits, picked = [], []
        for i in np.argsort(-sims):
            if sims[i] < min_similarity or len(hits) == k:
                break
            if self.rows[i][0] in exclude or self.rows[i][1] == "exchange" and sims[i] < EXCHANGE_MIN:
                continue
            # One of each: the same question asked again scored 0.90-1.0 against its earlier copies (different
            # exchanges 0.55-0.81), and four copies of one answer only teach her to repeat it
            if any(float(self.vectors[i] @ self.vectors[j]) >= DUPLICATE for j in picked):
                continue
            picked.append(i)
            hits.append((float(sims[i]), *self.rows[i]))
        return hits

    def _delete(self, memory_id):
        self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        index = next(n for n, row in enumerate(self.rows) if row[0] == memory_id)
        del self.rows[index]
        self.vectors = np.delete(self.vectors, index, axis=0) if len(self.rows) else None
        self.db.commit()

    def forget(self, about, min_similarity=0.3):
        """Delete the remembered facts and skills that best match `about` (never the conversation log); returns
        their text"""
        doomed = [hit for hit in self.search(about, k=3, min_similarity=min_similarity) if hit[2] in ("fact", "skill")]
        for _, memory_id, *_ in doomed:
            self._delete(memory_id)
        return [text for *_, text, _ in doomed]

    def save_skill(self, name, when, steps):
        """Store how to do a multi-step job, by name, replacing a skill saved under the same name; its id, or None"""
        name = " ".join(name.split())
        for memory_id, kind, text, _ in list(self.rows):
            if kind == "skill" and skill_name(text) == name.lower():
                self._delete(memory_id)
        listed = " ".join(f"{n}. {step.strip().rstrip('.')}." for n, step in enumerate(steps, 1) if step.strip())
        return self.add(f'Skill "{name}" - use when {when.strip().rstrip(".")}. Steps: {listed}', "skill")

    def skills(self):
        return [text for _, kind, text, _ in self.rows if kind == "skill"]

    def tools(self):
        @tool
        def remember(fact: str) -> str:
            """Save a fact to remember long-term, e.g. "Aalok's sister Priya's birthday is June 3".
            Use when the user asks you to remember something, or tells you something worth keeping.
            Write it as a standalone sentence that makes sense on its own later"""
            return "Remembered." if self.add(fact, "fact") else "Couldn't save that - memory is unavailable."

        @tool
        def forget(about: str) -> str:
            """Delete remembered facts or skills about something, when the user asks you to forget it"""
            gone = self.forget(about)
            return f"Forgot: {'; '.join(gone)}" if gone else "Nothing like that was remembered."

        @tool
        def save_skill(name: str, when: str, steps: list[str]) -> str:
            """Save how to do a multi-step job as a named skill, to repeat it later. Use when the user asks you to
            remember how to do something ("save that as my evening routine"). name: short, e.g. "evening routine".
            when: when to use it, e.g. "Aalok asks for his evening routine". steps: plain instructions you could
            follow again, e.g. ["Look up tonight's weather in Mumbai", "Play a chill playlist that suits it",
            "Set the Spotify volume to 30 percent"]. Saving under an existing name replaces that skill"""
            if not steps:
                return "Not saved: a skill needs its steps"
            saved = self.save_skill(name, when, steps)
            return f'Saved the skill "{name}" ({len(steps)} steps).' if saved else "Couldn't save that - memory is unavailable."

        @tool
        def list_skills() -> str:
            """The skills (saved routines) you know, for "what routines do you know?" """
            names = [skill_name(text) for text in self.skills()]
            return "Saved skills: " + ", ".join(names) if names else "No skills saved yet."

        return [remember, forget, save_skill, list_skills]


def skill_name(text):
    """The name a skill was saved under, lowercased"""
    match = re.match(r'Skill "(.+?)"', text)
    return match.group(1).lower() if match else ""


def note(hits):
    """The memories as a short block appended to the user's message - after the stable system prompt,
    so llama-server's prompt cache still works"""
    if not hits:
        return ""
    lines = [f"- ({datetime.fromtimestamp(created):%d %b %Y}) {text}" for _, _, _, text, created in hits]
    return "\n[From memory - use only if relevant, don't read out]\n" + "\n".join(lines)
