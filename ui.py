"""NIA control panel: edit settings, start/stop the LLM server and the assistant. Run: python ui.py"""
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import settings
from agent.chat import llm_server_command, llm_up
from utils.logger import logging

logger = logging.getLogger("ui")
ROOT = Path(__file__).resolve().parent
NEW_WINDOW = subprocess.CREATE_NEW_CONSOLE  # each process gets its own console, so its logs stay visible
# python.exe even when this panel runs under pythonw.exe, which has no console or stdin for its children
PYTHON = str(Path(sys.executable).with_name("python.exe"))

# (key, label, widget) - a None key starts a new section
FIELDS = [
    (None, "LLM server  (GPU - restart the server to apply)", None),
    ("bonsai_dir", "Bonsai folder", "entry"),
    ("model_file", "Model file", "entry"),
    ("port", "Port", "entry"),
    ("context", "Context (tokens)", "entry"),
    ("cache_8bit", "8-bit context cache (saves ~0.5 GB VRAM)", "check"),
    ("thinking", "Thinking", "check"),
    ("temperature", "Temperature", "entry"),
    ("top_p", "Top-p", "entry"),
    ("top_k", "Top-k", "entry"),
    ("presence_penalty", "Presence penalty", "entry"),
    (None, "Assistant  (CPU - restart the assistant to apply)", None),
    ("voice", "Voice", "voice"),
    ("voice_volume", "NIA's voice volume (0.1-1, 1 = full)", "entry"),
    ("stt_model", "Speech-to-text model", "stt"),
    ("wake_phrase", "Wake phrase", "entry"),
    ("wake_threshold", "Wake match threshold (0-1, lower wakes easier)", "entry"),
    ("greeting", "Greeting", "entry"),
    ("duck_level", "Other apps' volume while NIA speaks (0-1, 1 = off)", "entry"),
    ("session_timeout", "Session timeout (s)", "entry"),
    ("music_moods", "Random music picks (comma-separated)", "entry"),
    ("approval_model", "Approval model (Ollama; empty = always ask)", "entry"),
    ("approval_threshold", "Approval threshold (0-1, lower asks more)", "entry"),
]


def voices():
    """Every English voice Moonshine offers, downloaded ones first"""
    try:
        from moonshine_voice import list_tts_voices
        v = list_tts_voices("en_us")
        return v["present"] + v["downloadable"]
    except Exception:  # catalog unavailable - the field still takes any voice id
        return [settings.DEFAULTS["voice"]]


class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIA control panel")
        self.resizable(False, False)
        self.assistant = None  # the main.py process, if started from here
        self.vars = {}

        form = ttk.Frame(self, padding=12)
        form.grid(sticky="nsew")
        current = settings.load()
        for row, (key, label, widget) in enumerate(FIELDS):
            if key is None:
                ttk.Label(form, text=label, font=("Segoe UI", 10, "bold")).grid(
                    row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
                continue
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            if widget == "check":
                var = tk.BooleanVar(value=current[key])
                ttk.Checkbutton(form, variable=var).grid(row=row, column=1, sticky="w")
            else:
                var = tk.StringVar(value=str(current[key]))
                if widget == "voice":
                    ttk.Combobox(form, textvariable=var, values=voices(), width=38).grid(row=row, column=1, sticky="w")
                elif widget == "stt":
                    ttk.Combobox(form, textvariable=var, values=["tiny", "small", "medium"],
                                 state="readonly", width=38).grid(row=row, column=1, sticky="w")
                else:
                    ttk.Entry(form, textvariable=var, width=41).grid(row=row, column=1, sticky="w")
            self.vars[key] = var

        buttons = ttk.Frame(self, padding=(12, 0, 12, 12))
        buttons.grid(sticky="ew")
        for col, (text, command) in enumerate([
            ("Save", self.save), ("Reset to defaults", self.reset),
            ("Start server", self.start_server), ("Stop server", self.stop_server),
            ("Start assistant", self.start_assistant), ("Stop assistant", self.stop_assistant),
            ("Text chat", self.text_chat), ("Spotify login", self.spotify_login),
        ]):
            ttk.Button(buttons, text=text, command=command).grid(row=col // 2, column=col % 2, sticky="ew", padx=2, pady=2)
        buttons.columnconfigure((0, 1), weight=1)

        self.status = ttk.Label(self, padding=(12, 0, 12, 12))
        self.status.grid(sticky="w")
        self.refresh_status()

    def collect(self):
        """Form values converted to each default's type; None (after an error box) if one doesn't parse"""
        values = {}
        for key, var in self.vars.items():
            kind = type(settings.DEFAULTS[key])
            try:
                values[key] = var.get() if kind is bool else kind(var.get().strip())
            except ValueError:
                messagebox.showerror("Invalid setting", f"{key} must be a {'whole ' if kind is int else ''}number, got {var.get()!r}")
                return None
        for key, label, low, high in (("wake_threshold", "Wake match threshold", 0, 1),
                                      ("duck_level", "Other apps' volume", 0, 1),
                                      ("approval_threshold", "Approval threshold", 0, 1),
                                      ("voice_volume", "NIA's voice volume", 0.1, 1)):
            if not low <= values[key] <= high:
                messagebox.showerror("Invalid setting", f"{label} must be between {low} and {high}")
                return None
        if not values["wake_phrase"].strip():
            messagebox.showerror("Invalid setting", "Wake phrase can't be empty")
            return None
        if not any(m.strip() for m in values["music_moods"].split(",")):
            messagebox.showerror("Invalid setting", "Give at least one random music pick, e.g. lofi beats")
            return None
        return values

    def report_callback_exception(self, exc_type, exc, tb):
        """tkinter swallows button-handler errors; log them and show them instead"""
        logger.error("Control panel error", exc_info=(exc_type, exc, tb))
        messagebox.showerror("Error", f"{exc_type.__name__}: {exc}\n\nDetails in logs/nia.log")

    def save(self):
        values = self.collect()
        if values is not None:
            settings.save(values)
            logger.info(f"Settings saved: {values}")
        return values

    def reset(self):
        for key, var in self.vars.items():
            var.set(settings.DEFAULTS[key] if isinstance(var, tk.BooleanVar) else str(settings.DEFAULTS[key]))
        self.save()

    def start_server(self):
        values = self.save()
        if values is None:
            return
        if llm_up(values["port"]):
            messagebox.showinfo("LLM server", "Already running. Stop it first to apply new server settings.")
            return
        logger.info("Starting LLM server from the control panel")
        subprocess.Popen(llm_server_command(values), creationflags=NEW_WINDOW)

    def stop_server(self):
        logger.info("Stopping LLM server from the control panel")
        subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"], capture_output=True)

    def start_assistant(self):
        if self.assistant and self.assistant.poll() is None:
            messagebox.showinfo("Assistant", "Already running. Stop it first to apply new assistant settings.")
            return
        if self.save() is not None:  # main.py starts the server itself if it is down
            logger.info("Starting assistant from the control panel")
            env = {**os.environ, "NIA_PAUSE_ON_CRASH": "1"}  # keep its window open on a crash
            self.assistant = subprocess.Popen([PYTHON, "main.py"], cwd=ROOT, env=env, creationflags=NEW_WINDOW)

    def stop_assistant(self):
        if self.assistant and self.assistant.poll() is None:
            logger.info("Stopping assistant from the control panel")
            self.assistant.terminate()

    def text_chat(self):
        if self.save() is not None:
            logger.info("Opening text chat from the control panel")
            subprocess.Popen([PYTHON, "test.py"], cwd=ROOT, creationflags=NEW_WINDOW)

    def spotify_login(self):
        """Fresh Spotify login, for when NIA reports 'Refresh token expired'"""
        logger.info("Opening Spotify login from the control panel")
        subprocess.Popen([PYTHON, "-m", "agent.action_controller"], cwd=ROOT, creationflags=NEW_WINDOW)

    def refresh_status(self):
        port = settings.load()["port"]
        server = "running" if llm_up(port) else "stopped"
        assistant = "running" if self.assistant and self.assistant.poll() is None else "stopped"
        self.status.config(text=f"LLM server (port {port}): {server}     Assistant: {assistant}")
        self.after(2000, self.refresh_status)


if __name__ == "__main__":
    ControlPanel().mainloop()
