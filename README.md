# NIA — Next-gen Intelligence Agent

A Jarvis-style voice assistant that runs entirely on your own machine. Wake phrase,
speech recognition, reasoning, and speech synthesis are all local — nothing is sent
to a cloud API, and no accounts or keys are needed. The only network calls are to
Spotify, and only when you ask for music.

Say **"Hey Nia"**, then talk — or say it all at once: "Hey Nia, play some lofi".

## How it works

```
  mic ──▶ Moonshine ──────────────────▶ LangGraph agent ◀──▶ spotipy
          (wake phrase + streaming       (Bonsai 2 27B on
           STT + VAD, CPU)                llama-server, GPU)
  speaker ◀── Kokoro TTS (CPU) ◀────────────────┘
```

Everything but the LLM runs on the CPU, so the GPU is left to Bonsai alone.

One Moonshine listener runs all the time. Its voice-activity detection only runs the
model while someone is speaking, so idle cost is under 1% CPU. Each finished utterance
drives a two-state machine ([voice_assistant/voice_assistant.py](voice_assistant/voice_assistant.py)):

| State | What happens |
|---|---|
| `LISTENING` | Utterances are checked for the wake phrase and otherwise ignored (not logged). Anything said after the phrase in the same breath is run as a command. |
| `COMMAND_MODE` | Says "Yes, sir?" (unless you already gave a command), then each utterance goes to the agent: reason → act → speak. Returns to `LISTENING` after 60s with no exchange. |

The wake phrase is matched fuzzily against the start of each utterance, because the
recogniser hears "Hey Nia" as "Hey, Nia." or with a stray word in front. The match
threshold (default 0.8, in the control panel) is the knob: lower wakes more easily but
false-triggers more — "Hey Mia" already scores 0.83. The mic is muted while NIA thinks
and speaks, so she doesn't transcribe herself. While she speaks, every other app (Spotify,
browser, games) is turned down to 30% of its volume in the Windows mixer and restored
afterwards — "Other apps' volume while NIA speaks" in the control panel; 1 turns it off.
Her own voice volume is a tool too: "speak up", "you're too loud", "talk at 50 percent"
(10-100%; more would only clip). It's saved, so it survives restarts.

The agent is a small LangGraph loop in [agent/chat.py](agent/chat.py): the model
either answers, or calls Spotify tools; tool results (and errors) go back to the model,
which then answers in a plain sentence. So you hear "Playing Highway to Hell by AC/DC"
rather than a raw API response. History keeps the last 20 messages, cut on whole turns.

## Requirements

- **Python 3.11** (tested)
- **NVIDIA GPU with 8 GB VRAM** for the LLM. Bonsai 2 27B (`PTQ1_0`, 5.95 GB) runs fully on
  the GPU at ~35 tok/s on an RTX 4060 Laptop
- **[PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp/releases)**, since
  stock llama.cpp and Ollama can't load Bonsai's ternary format, plus
  [`Ternary-Bonsai-2-27B-PTQ1_0.gguf`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
  Both are expected under `D:\bonsai` (the fork in `llama-prism\`) — change "Bonsai folder"
  in the control panel if yours live elsewhere
- A working microphone and speaker
- A [Spotify Developer](https://developer.spotify.com/dashboard) app, for music control

## Setup

```bash
git clone https://github.com/alowkii/nia
cd nia

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

The Moonshine speech model and the Kokoro voice download on first run.

Create a `.env` in the project root:

```ini
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

The easiest way is the control panel:

```bash
python ui.py
```

It edits every setting (model, context, thinking, sampling, voice, speech model, wake
phrase and match threshold, greeting, timeout) and starts or stops the LLM server, the assistant, and the
text chat, each in its own console window. Settings are saved to `settings.json`; the
defaults live in [settings.py](settings.py). Server settings apply when the server
restarts, assistant settings when the assistant restarts.

Or run the assistant directly:

```bash
python main.py
```

If the LLM server isn't running, this starts it in its own window and waits for it
(~20 s). Wait for `Listening for 'hey nia'...`, then say "Hey Nia".

For a keyboard-driven session with no audio stack at all — useful for iterating on
prompts — use [test.py](test.py):

```bash
python test.py
```

## Logs

Everything lands in `logs/`:

| File | What's in it |
|---|---|
| `nia.log` | Each turn — what was heard (and transcription latency), tool calls with arguments and results, the reply, LLM and speech timings — plus startup, control-panel actions, and every crash with its full traceback. Rotates at 5 MB, keeping 3 old files. |
| `llama-server.log` | The LLM server's own output: model loading, per-request token speeds, errors. Rewritten each time the server starts. |

When the assistant is started from the control panel, its window stays open after a
crash so you can read the error.

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
python test_voice.py
```

[test_voice.py](test_voice.py) checks the fuzzy wake-phrase matcher against real Moonshine
transcripts — what should wake her, what shouldn't, and where the threshold cuts.

[test_agent.py](test_agent.py) runs the real agent graph offline with a scripted
model and a fake Spotify client, so it needs no network, no mic, and no LLM server. It
checks that tool calls reach the right controller method, that Spotify errors go back
to the model instead of crashing, and that history is trimmed on whole turns.

## Layout

```
main.py                             entry point
ui.py                               control panel: settings, start/stop everything
settings.py                         setting defaults (overrides in settings.json)
voice_assistant/voice_assistant.py  state machine, STT, TTS
run_bonsai.bat                      starts the LLM server (llama-server + Bonsai)
agent/chat.py                       LangGraph agent and Spotify tools
agent/action_controller.py          Spotify operations
agent/prompts/                      system prompt
preprocess_voice.py                 builds a speaker embedding from a voice sample
test_agent.py, test_voice.py         offline tests
test.py                             text-only REPL
```

## Known limitations

- **Speaker verification is disabled.** The code to check that a command came from
  your voice is present but commented out in `voice_assistant.py` — it added too much
  latency per command. `preprocess_voice.py` still generates the reference embedding.
  Anyone within earshot can currently issue commands.
- **Replies take ~3–4 s** from the end of your sentence: Bonsai reads the prompt and
  tools, calls a tool, then words the result. Kokoro adds ~1 s before speech starts;
  set `VOICE = "piper_en_US-lessac-medium"` for ~0.2 s with a more robotic voice.
- **Spotify needs an active device.** Playback commands fail if Spotify isn't already
  open somewhere.

## License

MIT — see [LICENSE](LICENSE).
