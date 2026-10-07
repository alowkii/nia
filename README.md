# NIA — Next-gen Intelligence Agent

A Jarvis-style voice assistant that runs entirely on your own machine. Wake word,
speech recognition, reasoning, and speech synthesis are all local — nothing is sent
to a cloud API. The only network calls are to Spotify, and only when you ask for music.

Say **"Hey Nia"**, then talk.

## How it works

```
  mic ──▶ Porcupine ──▶ sounddevice ──▶ Whisper ──▶ LangGraph agent ◀──▶ spotipy
        (wake word)      (record +      (tiny,     (Ollama, tool calls)
                          silence VAD)   local)            │
        speaker ◀── VITS TTS ◀─────────────────────────────┘
```

A three-state machine drives it ([voice_assistant/voice_assistant.py](voice_assistant/voice_assistant.py)):

| State | What happens |
|---|---|
| `LISTENING` | Porcupine scans mic frames for the wake word. Cheap; this is the idle state. |
| `WAKE_DETECTED` | Flushes buffered audio so the wake word itself isn't transcribed as a command. |
| `COMMAND_MODE` | Greets you, then loops: record → transcribe → reason → act → speak. Returns to `LISTENING` after 60s with no exchange. |

Recording ends on its own. The noise floor is measured before each capture and
speech is considered finished after 1.0s below that threshold, with a 10s hard cap.

The agent is a small LangGraph loop in [agent/chat.py](agent/chat.py): the model
either answers, or calls Spotify tools; tool results (and errors) go back to the model,
which then answers in a plain sentence. So you hear "Playing Highway to Hell by AC/DC"
rather than a raw API response. History keeps the last 20 messages, cut on whole turns.

## Requirements

- **Python 3.11** — Coqui TTS 0.22 does not support 3.12+
- **[Ollama](https://ollama.com)** running locally, with `qwen3:8b` pulled
- A working microphone and speaker
- **NVIDIA GPU (optional)** — CUDA is used when available; falls back to CPU automatically
- A [Picovoice](https://picovoice.ai) access key (free tier is fine)
- A [Spotify Developer](https://developer.spotify.com/dashboard) app, for music control

Note: the bundled wake-word model is `wake_word/Hey-Nia_en_windows_v3_0_0.ppn`, which
is **Windows-only**. Porcupine models are platform-specific — on Linux or macOS,
generate a matching `.ppn` from the Picovoice console.

## Setup

```bash
git clone https://github.com/alowkii/nia
cd nia

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

ollama pull qwen3:8b
```

Create a `.env` in the project root:

```ini
PICOVOICE_ACCESS_KEY=your_picovoice_key
SPOTIFY_ID=your_spotify_client_id
SPOTIFY_SECRET=your_spotify_client_secret
AUTHOR=your_name
```

`AUTHOR` is just what the assistant calls you. In your Spotify app settings, add the
redirect URI that matches [`agent/action_controller.py`](agent/action_controller.py) —
it currently expects `https://aalokpandit.netlify.app/`, so change it to your own.
The first music command opens a browser once for OAuth and caches the token in
`.spotify_cache`.

## Running

```bash
python main.py
```

Wait for `Listening for wake words...`, then say "Hey Nia".

For a keyboard-driven session with no audio stack at all — useful for iterating on
prompts — use [test.py](test.py):

```bash
python test.py
```

## What it can do

Spotify is the only integration so far. Ask in plain language; the model maps it to
one of these:

| | |
|---|---|
| **Playback** | play a track, playlist, or album; pause; resume; next; previous |
| **Queue** | add a track to the queue |
| **Volume** | set to a percentage; step up; step down |
| **Modes** | shuffle; repeat track; repeat playlist; repeat off |
| **Query** | what's playing right now |

Adding a new capability means a method on the controller and a `@tool` function in
`_spotify_tools()` in [`agent/chat.py`](agent/chat.py). The tool's signature and
docstring are what the model sees, so the docstring is the prompt.

## Tests

```bash
python test_agent.py
```

[test_agent.py](test_agent.py) runs the real agent graph offline with a scripted
model and a fake Spotify client, so it needs no network, no mic, and no Ollama. It
checks that tool calls reach the right controller method, that Spotify errors go back
to the model instead of crashing, and that history is trimmed on whole turns.

## Layout

```
main.py                             entry point
voice_assistant/voice_assistant.py  state machine, audio capture, STT, TTS
agent/chat.py                       LangGraph agent and Spotify tools
agent/action_controller.py          Spotify operations
agent/prompts/                      system prompt
wake_word/                          Porcupine .ppn model
preprocess_voice.py                 builds a speaker embedding from a voice sample
test_agent.py                       offline tests
test.py                             text-only REPL
```

## Known limitations

- **Speaker verification is disabled.** The code to check that a command came from
  your voice is present but commented out in `voice_assistant.py` — it added too much
  latency per command. `preprocess_voice.py` still generates the reference embedding.
  Anyone within earshot can currently issue commands.
- **Whisper `tiny`** is chosen for speed, not accuracy. Unusual track names get
  mangled. Moving to `base` or `small` is a one-word change with a latency cost.
- **Spotify needs an active device.** Playback commands fail if Spotify isn't already
  open somewhere.

## License

MIT — see [LICENSE](LICENSE).
