"""NIA in one window. Run: python nia.py  (or pythonw nia.py for no console at all)

Opens the HUD (hud/hud.html) in an Edge app window and runs everything behind it, hidden: the LLM server
(llama-server + Bonsai, GPU) and the assistant (main.py: mic, speech, agent) as child processes. The window
shows what NIA is doing, takes typed commands, and holds every setting. Closing it shuts everything down.
Starting it again while NIA runs just opens another window onto her.
"""
import json
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing.connection import Listener
from pathlib import Path

import psutil
from dotenv import load_dotenv

import settings
from agent.server import llm_server_command, llm_up
from utils import mics
from utils.logger import LOG_DIR, logging

load_dotenv()
logger = logging.getLogger("nia")

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "hud" / "hud.html"
PORT = 8765  # 8080 is httpd's, 8081 the LLM server's
PROFILE = ROOT / ".hud-browser"  # the window's own browser profile
HIDDEN = subprocess.CREATE_NO_WINDOW
PYTHON = str(Path(sys.executable).with_name("python.exe"))  # pythonw has no stdout for the child to log to
WINDOW_GONE = 5  # seconds with no page connected before NIA takes the window as closed

# The settings panel: (section, [(key, label, kind)]); kind is text | number | check | voice | stt
FIELDS = [
    ("LLM server · saving restarts it", [
        ("bonsai_dir", "Bonsai folder", "text"), ("model_file", "Model file", "text"), ("port", "Port", "number"),
        ("context", "Context (tokens)", "number"), ("cache_8bit", "8-bit context cache (saves ~0.5 GB VRAM)", "check"),
        ("thinking", "Thinking", "check"), ("temperature", "Temperature", "number"), ("top_p", "Top-p", "number"),
        ("top_k", "Top-k", "number"), ("presence_penalty", "Presence penalty", "number")]),
    ("Voice", [
        ("voice", "Voice", "voice"), ("voice_speed", "Speaking speed (0.5-2, 1 = natural)", "number"),
        ("voice_volume", "Voice volume (0.1-1)", "number"),
        ("max_spoken_words", "Words spoken before “Shall I go on?”", "number"),
        ("greeting", "Greetings (| between them)", "text"),
        ("duck_level", "Other apps' volume while she speaks (0-1, 1 = off)", "number")]),
    ("Listening", [
        ("mic_device", "Microphone", "mic"), ("stt_model", "Speech-to-text model", "stt"),
        ("wake_phrase", "Wake phrase", "text"),
        ("wake_threshold", "Wake match threshold (0-1, lower wakes easier)", "number"),
        ("session_timeout", "Seconds awake with no command", "number")]),
    ("Memory and approvals", [
        ("embedding_model", "Memory embedding model (Ollama; empty = no memory)", "text"),
        ("memory_results", "Memories recalled per turn", "number"),
        ("memory_min_similarity", "Memory match threshold (0-1)", "number"),
        ("history_turns", "Recent exchanges sent to the LLM", "number"),
        ("approval_model", "Approval model (Ollama; empty = always ask)", "text"),
        ("approval_threshold", "Approval threshold (0-1, lower asks more)", "number")]),
    ("Music", [("music_moods", "Random music picks (comma-separated)", "text")]),
    ("Ask Claude", [("claude_model", "Claude model for questions (sonnet, opus, haiku; empty = default)", "text"),
                    ("claude_effort", "Effort: low is fastest (~20 s); medium or high checks more, slower", "text")]),
]
SERVER_KEYS = {key for key, _, _ in FIELDS[0][1]}
# Read by the running assistant each time they're used, so they change without a restart
LIVE_KEYS = {"claude_effort", "claude_model", "mic_device", "voice_speed", "voice_volume", "max_spoken_words", "greeting", "duck_level", "wake_threshold",
             "session_timeout", "wake_word", "push_to_talk", "spoken_replies", "hud_theme"}
VOICES = ["kokoro_bm_fable", "kokoro_bm_george", "kokoro_af_heart"]
STT_MODELS = ["tiny", "small", "medium"]


def config():
    """What the page needs to know beyond NIA's state"""
    s = settings.load()
    return {"author": os.getenv("AUTHOR", "you"), "wake_phrase": s["wake_phrase"], "voice": s["voice"],
            "voice_speed": s["voice_speed"], "theme": s["hud_theme"], "wake_word": s["wake_word"],
            "push_to_talk": s["push_to_talk"], "spoken_replies": s["spoken_replies"], "mic_device": s["mic_device"]}


class Hub:
    """The page's live view: each update goes to every open page, and the latest of each kind is kept for a
    page that opens (or reloads) later"""

    def __init__(self):
        self.pages = set()
        self.lock = threading.Lock()
        self.latest = {"state": "booting", "server": "loading"}
        self.last_seen = None  # when a page was last connected; None until the first one

    def send(self, **message):
        with self.lock:
            self.latest.update({k: v for k, v in message.items() if k not in ("envelope", "fps", "partial")})
            for page in self.pages:
                page.put(message)


def bonsai_servers():
    """NIA's llama-server processes, answering or not - by their path, since Ollama runs its own llama-server.exe"""
    exe = Path(llm_server_command()[0]).resolve()
    found = []
    for proc in psutil.process_iter(["name", "exe"]):
        try:
            if proc.info["name"] == "llama-server.exe" and proc.info["exe"] and Path(proc.info["exe"]).resolve() == exe:
                found.append(proc)
        except (psutil.Error, OSError):
            pass
    return found


def ensure_ollama():
    """Memory and auto-approval run on Ollama: start its tray app (as at login) if it isn't up"""
    s = settings.load()
    if not (s["embedding_model"] or s["approval_model"]):
        return
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2).close()
        return
    except OSError:
        pass
    app = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama app.exe"
    if app.exists():
        logger.info("Starting Ollama (memory and auto-approval need it)")
        subprocess.Popen([str(app)], creationflags=HIDDEN)
    else:
        logger.warning("Ollama isn't running or installed - no memory, and every PC command will ask first")


class Server:
    """llama-server, started hidden unless one is already up; stopped on exit only if NIA started it"""

    def __init__(self, hub):
        self.hub, self.proc = hub, None

    def start(self):
        s = settings.load()
        if llm_up(s["port"]):
            self.hub.send(server="ready")
            return
        self.hub.send(server="loading")
        if bonsai_servers():  # one is still loading (or stuck): a second copy would need the GPU twice over
            logger.info("An LLM server is already starting - waiting for it")
            threading.Thread(target=self._wait, args=(None, s["port"]), daemon=True).start()
            return
        logger.info(f"Starting LLM server: {' '.join(llm_server_command(s))}")
        self.proc = subprocess.Popen(llm_server_command(s), creationflags=HIDDEN,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # it logs to its own file
        threading.Thread(target=self._wait, args=(self.proc, s["port"]), daemon=True).start()

    def _wait(self, proc, port, timeout=180):
        """Until the server answers; proc is None when waiting on a server NIA didn't start"""
        end = time.time() + timeout
        while (proc.poll() is None) if proc else time.time() < end:
            if llm_up(port):
                logger.info("LLM server ready")
                self.hub.send(server="ready")
                return
            time.sleep(1)
        if proc is self.proc:
            logger.error("LLM server didn't come up - see logs/llama-server.log; Restart server in the settings")
            self.hub.send(server="down")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(10)

    def restart(self):
        """To apply new server settings - including a server NIA didn't start"""
        self.stop()
        for proc in bonsai_servers():  # never Ollama's
            proc.kill()
        time.sleep(1)  # let the GPU memory go
        self.start()


class Assistant:
    """main.py as a hidden child process, joined to the hub over an authenticated local connection"""

    def __init__(self, hub, server=None):
        self.hub, self.server, self.proc, self.conn = hub, server, None, None  # server: restarted when she asks
        self.stopping = False

    def start(self):
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        env = {**os.environ, "NIA_HUD": f"{listener.address[1]}:{key.hex()}",
               "PYTHONIOENCODING": "utf-8"}  # its log file, not Windows' old code page ("—" came out as "�")
        console = open(LOG_DIR / "assistant.out.log", "ab")  # native crashes print here, not to nia.log
        self.stopping = False
        self.proc = subprocess.Popen([PYTHON, "main.py"], cwd=ROOT, env=env, creationflags=HIDDEN,
                                     stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
        console.close()
        self.hub.send(state="booting", error=None)
        threading.Thread(target=self._relay, args=(listener,), daemon=True).start()
        threading.Thread(target=self._watch, args=(listener, self.proc), daemon=True).start()

    def _relay(self, listener):
        try:
            self.conn = listener.accept()
            while True:
                if message := self.handle(self.conn.recv()):
                    self.hub.send(**message)
        except (EOFError, OSError):  # the assistant exited, or never connected
            pass

    def handle(self, message):
        """A restart she asked for (restart_myself) is carried out; everything else is for the page"""
        what = message.pop("restart", None)
        if what:
            logger.info(f"She asked to be restarted ({what})")

            def restart():
                if what == "both" and self.server:
                    self.server.restart()  # the assistant waits for it as it starts
                self.restart()
            threading.Thread(target=restart, daemon=True).start()
        return message

    def _watch(self, listener, proc):
        code = proc.wait()
        listener.close()  # frees _relay if the assistant died before connecting
        if proc is self.proc:  # not one that's being replaced
            self.conn = None
            crashed = code != 0 and not self.stopping
            if crashed:
                logger.error(f"Assistant exited with code {code} - see logs/nia.log and logs/assistant.out.log")
            self.hub.send(state="offline", error="crashed" if crashed else None)

    def send(self, message):
        try:
            if self.conn:
                self.conn.send(message)
        except OSError:
            pass

    def stop(self, timeout=10):
        self.stopping = True
        if self.proc and self.proc.poll() is None:
            self.send({"type": "quit"})
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def restart(self):
        self.stop()
        self.start()


def apply(changed, values, assistant, server, hub):
    """A saved change: live ones go straight to the assistant, the rest restart what reads them at start"""
    live = {key: values[key] for key in changed if key in LIVE_KEYS}
    if live:
        assistant.send({"type": "settings", "values": live})
    hub.send(config=config())
    restarting = []
    if changed & SERVER_KEYS:
        restarting.append("server")
        threading.Thread(target=server.restart, daemon=True).start()
    if changed - LIVE_KEYS:  # the assistant also reads the server's port and context at start
        restarting.append("assistant")
        threading.Thread(target=assistant.restart, daemon=True).start()
    return restarting


def handler(hub, assistant, server, done):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # requests aren't worth a log line
            pass

        def reply(self, code, body=b"", kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/settings":
                s = settings.load()
                voices = VOICES + ([s["voice"]] if s["voice"] not in VOICES else [])
                self.reply(200, json.dumps({"values": s, "fields": FIELDS, "voices": voices, "stt_models": STT_MODELS,
                                            "mics": mics.names(refresh=True)}).encode())
            elif self.path == "/mics":  # for the picker by the mic button
                self.reply(200, json.dumps({"mics": mics.names(refresh=True),
                                            "current": settings.load()["mic_device"]}).encode())
            elif self.path == "/events":
                self.events()
            else:
                self.reply(404)

        def events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            page = queue.Queue()
            with hub.lock:
                page.put({**hub.latest, "config": config()})
                hub.pages.add(page)
            try:
                while True:
                    try:
                        message = page.get(timeout=2)
                        self.wfile.write(f"data: {json.dumps(message)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": still here\n\n")  # a closed window shows up as a failed write
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                with hub.lock:
                    hub.pages.discard(page)
                    hub.last_seen = time.time()

        def do_POST(self):
            # A web page can't send this header to localhost without a CORS preflight, which is never answered
            if self.headers.get("X-NIA") != "1":
                self.reply(403)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/command":
                assistant.send({"type": "text", "text": str(body.get("text", ""))[:2000]})
            elif self.path == "/mic":
                action = body.get("action")
                if action == "restart":
                    threading.Thread(target=assistant.restart, daemon=True).start()
                elif action in ("wake", "sleep", "ptt_down", "ptt_up"):
                    assistant.send({"type": action})
            elif self.path == "/settings":
                before = settings.load()
                values, error = settings.parse(body)
                if error:
                    self.reply(400, json.dumps({"error": error}).encode())
                    return
                settings.save(values)
                changed = {key for key in values if values[key] != before[key]}
                logger.info(f"Settings changed: { {key: values[key] for key in changed} }")
                self.reply(200, json.dumps({"restarting": apply(changed, values, assistant, server, hub)}).encode())
                return
            elif self.path == "/server/restart":
                threading.Thread(target=server.restart, daemon=True).start()
            elif self.path == "/spotify/login":
                from agent.action_controller import oauth
                url = oauth().get_authorize_url()
                webbrowser.open(url)
                self.reply(200, json.dumps({"url": url}).encode())
                return
            elif self.path == "/spotify/code":
                from agent.action_controller import oauth
                try:
                    login = oauth()
                    login.get_access_token(login.parse_response_code(str(body.get("url", "")).strip()),
                                           as_dict=False, check_cache=False)
                except Exception as e:  # a wrong or stale address: say so in the panel
                    self.reply(400, json.dumps({"error": f"Spotify didn't accept that: {e}"}).encode())
                    return
                logger.info("Logged in to Spotify")
            elif self.path == "/quit":
                done.set()
            else:
                self.reply(404)
                return
            self.reply(204)

    return Handler


class HudServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # on Windows it would let a second NIA bind the same port instead of failing


def open_window(url):
    """An Edge app window (no browser bars) with its own profile; a normal tab if there's no Edge or Chrome"""
    browser = next((p for p in (shutil.which("msedge"),
                                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe") if p and Path(p).exists()),
                   None)
    if not browser:
        webbrowser.open(url)
        return None
    return subprocess.Popen([browser, f"--app={url}", "--start-maximized", f"--user-data-dir={PROFILE}",
                             "--no-first-run", "--no-default-browser-check"])


def main():
    url = f"http://127.0.0.1:{PORT}"
    hub, done = Hub(), threading.Event()
    server = Server(hub)
    assistant = Assistant(hub, server)
    try:
        httpd = HudServer(("127.0.0.1", PORT), handler(hub, assistant, server, done))
    except OSError:  # NIA is already running: just show her
        logger.info("NIA is already running - opening another window onto her")
        open_window(url)
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    logger.info(f"NIA's window: {url}")
    window = open_window(url)
    ensure_ollama()
    server.start()
    assistant.start()

    def window_closed():  # no page connected for a while: the window was closed
        while not done.is_set():
            time.sleep(1)
            with hub.lock:
                gone = not hub.pages and hub.last_seen and time.time() - hub.last_seen > WINDOW_GONE
            if gone:
                logger.info("Window closed - shutting down")
                done.set()
    threading.Thread(target=window_closed, daemon=True).start()

    try:
        while not done.wait(0.5):  # a plain wait() can't be interrupted by Ctrl+C on Windows
            pass
    except KeyboardInterrupt:
        logger.info("Stopping (Ctrl+C)")
    assistant.stop()
    server.stop()
    if window and window.poll() is None:
        window.terminate()
    httpd.shutdown()


if __name__ == "__main__":
    main()
