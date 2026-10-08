"""The LLM server: llama-server (PrismML's fork) running Bonsai 2 27B on the GPU. Kept apart from chat.py so
nia.py can start and check it without loading the whole agent."""
import urllib.request
from pathlib import Path

import settings

# Server output also goes here, so it survives the server's console window closing (or never having one)
SERVER_LOG = Path(__file__).resolve().parent.parent / "logs" / "llama-server.log"


def llm_server_command(s=None):
    """llama-server running Bonsai 2 27B fully on the GPU, from settings"""
    s = s or settings.load()
    bonsai = Path(s["bonsai_dir"])
    SERVER_LOG.parent.mkdir(exist_ok=True)
    return [str(bonsai / "llama-prism" / "llama-server.exe"), "-m", str(bonsai / s["model_file"]),
            "-ngl", "99", "-fa", "on", "-c", str(s["context"]), "-np", "1",
            "--reasoning", "on" if s["thinking"] else "off",
            "--temp", str(s["temperature"]), "--top-p", str(s["top_p"]), "--top-k", str(s["top_k"]),
            "--presence-penalty", str(s["presence_penalty"]),
            "--host", "127.0.0.1", "--port", str(s["port"]), "--log-file", str(SERVER_LOG), "--log-colors", "off"] + \
        (["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"] if s["cache_8bit"] else [])


def llm_up(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except OSError:  # refused while starting, 503 while the model loads
        return False
