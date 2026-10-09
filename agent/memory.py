"""NIA's long-term memory: facts she's asked to remember, skills (how to do a multi-step job, saved by name to
repeat it), preferences she has learned by herself, and a log of past exchanges, kept in SQLite. Facts, skills and
exchanges are found by meaning (vector search), so each turn sends the LLM only the few that matter instead of the
whole history; preferences go with every turn.

Embeddings come from EmbeddingGemma on the CPU through Ollama. It was chosen against all-minilm,
granite-embedding and nomic-embed-text because it alone separated relevant memories (similarity >= 0.29)
from off-topic queries (<= 0.16), so unrelated turns get no memories at all. If Ollama is down, memory
switches itself off for that call and NIA carries on without it.
"""
import json
import logging
import re
import sqlite3
import threading
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
PREFERENCES_SHOWN = 15  # the most recent learned preferences, sent with every turn
# EmbeddingGemma's own prompts for stored documents and for searches
DOC_PREFIX, QUERY_PREFIX = "title: none | text: ", "task: search result | query: "


class Memory:
    def __init__(self, model, path=DB, embed=None):
        """model: the Ollama embedding model. embed: replaces Ollama (tests)."""
        self.model = model
        self._embed = embed or self._ollama_embed
        self.down_until = 0.0  # after a failed embedding, memory waits this long before trying again
        self.lock = threading.RLock()  # learning runs in the background, alongside a turn
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
        with self.lock:
            created = time.time()
            cursor = self.db.execute("INSERT INTO memories (kind, text, created, vector) VALUES (?, ?, ?, ?)",
                                     (kind, text, created, vector.astype(np.float32).tobytes()))
            self.db.commit()
            self.rows.append((cursor.lastrowid, kind, text, created))
            self.vectors = vector[None] if self.vectors is None else np.vstack([self.vectors, vector])
            return cursor.lastrowid

    def search(self, query, k=4, min_similarity=0.22, exclude=(), kinds=("fact", "skill", "exchange")):
        """[(similarity, id, kind, text, created)] most like the query, best first; nothing below min_similarity.
        Preferences aren't searched for by default: they go with every turn anyway"""
        vector = self._vector(query, query=True) if self.vectors is not None else None
        if vector is None:
            return []
        with self.lock:
            return self._search(vector, k, min_similarity, exclude, kinds)

    def _search(self, vector, k, min_similarity, exclude, kinds):
        sims = self.vectors @ vector
        hits, picked = [], []
        for i in np.argsort(-sims):
            if sims[i] < min_similarity or len(hits) == k:
                break
            row_id, kind = self.rows[i][0], self.rows[i][1]
            if row_id in exclude or kind not in kinds or kind == "exchange" and sims[i] < EXCHANGE_MIN:
                continue
            # One of each: the same question asked again scored 0.90-1.0 against its earlier copies (different
            # exchanges 0.55-0.81), and four copies of one answer only teach her to repeat it
            if any(float(self.vectors[i] @ self.vectors[j]) >= DUPLICATE for j in picked):
                continue
            picked.append(i)
            hits.append((float(sims[i]), *self.rows[i]))
        return hits

    def _delete(self, memory_id):
        with self.lock:
            self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            index = next(n for n, row in enumerate(self.rows) if row[0] == memory_id)
            del self.rows[index]
            self.vectors = np.delete(self.vectors, index, axis=0) if len(self.rows) else None
            self.db.commit()

    def forget(self, about, min_similarity=0.3):
        """Delete the remembered facts, skills and learned preferences that best match `about` (never the
        conversation log); returns their text"""
        doomed = self.search(about, k=3, min_similarity=min_similarity, kinds=("fact", "skill", "preference"))
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

    def preferences(self):
        """What she has learned about how the user likes things: [(id, text)], oldest first"""
        return [(memory_id, text) for memory_id, kind, text, _ in self.rows if kind == "preference"]

    def learn(self, manager, conversation):
        """Learn preferences from a finished conversation: manager (LangMem's memory manager) reads it with the
        preferences already known and says what to add, rewrite or drop, and that's applied here. Returns what
        changed, for the log"""
        from langmem.knowledge.extraction import Memory as Note  # LangMem's own type: as plain dicts, its
        known = dict(self.preferences())                             # edits to them failed to apply
        existing = [(str(memory_id), Note(content=text)) for memory_id, text in known.items()]
        changes = []
        for memory_id, content in manager.invoke({"messages": conversation, "existing": existing}):
            old = known.get(int(memory_id)) if str(memory_id).isdigit() else None
            if type(content).__name__ == "RemoveDoc":
                if old is not None:
                    self._delete(int(memory_id))
                    changes.append(f"dropped: {old}")
                continue
            text = content.get("content", "") if isinstance(content, dict) else getattr(content, "content", "")
            text = " ".join(str(text).split())
            if not text or text == old:
                continue
            if old is not None:
                self._delete(int(memory_id))
            if self.add(text, "preference") is not None:
                changes.append(f"changed: {old} -> {text}" if old else f"learned: {text}")
        return changes

    def tools(self):
        @tool
        def remember(fact: str) -> str:
            """Save a fact to remember long-term, e.g. "Aalok's sister Priya's birthday is June 3".
            Use when the user asks you to remember something, or tells you something worth keeping.
            Write it as a standalone sentence that makes sense on its own later"""
            return "Remembered." if self.add(fact, "fact") else "Couldn't save that - memory is unavailable."

        @tool
        def forget(about: str) -> str:
            """Delete remembered facts, skills or learned preferences about something, when the user asks you to
            forget it ("forget that I like jazz")"""
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


def preferences_note(preferences):
    """What she has learned about how the user likes things, for every turn - after the system prompt, like
    recalled memories, so llama-server's cache survives"""
    if not preferences:
        return ""
    lines = [f"- {text}" for _, text in preferences[-PREFERENCES_SHOWN:]]
    return ("\n[Learned about how they like things - follow it without being asked, mention it only if asked]\n"
            + "\n".join(lines))


def note(hits):
    """The memories as a short block appended to the user's message - after the stable system prompt,
    so llama-server's prompt cache still works"""
    if not hits:
        return ""
    lines = [f"- ({datetime.fromtimestamp(created):%d %b %Y}) {text}" for _, _, _, text, created in hits]
    return "\n[From memory - use only if relevant, don't read out]\n" + "\n".join(lines)
