# NIA — Next-gen Intelligence Agent

A Jarvis-style voice assistant that runs entirely on your own machine. Wake phrase,
speech recognition, reasoning, and speech synthesis are all local — nothing is sent
to a cloud AI API. The only network calls are to Spotify and YouTube, and only when
you ask for them; the only account needed is a Spotify developer app.

Say **"Hey Nia"**, then talk — or say it all at once: "Hey Nia, play some lofi".

## How it works

```
  mic ──▶ Moonshine ──────────────────▶ Deep Agent ◀──▶ Spotify, YouTube,
          (wake phrase + streaming       (Bonsai 2 27B on   the PC's files + shell
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
**Interrupting:** while she's thinking or talking, say **"Stop"**, **"Nia, stop"** or
**"Hey Nia"**. Her voice cuts off within a fraction of a second, and an unfinished turn
ends before its next step (steps already taken stay taken — an opened tab stays open).
"Hey Nia, pause the music" both stops her and does it. The mic stays live while she
works, but only a stop phrase at the start of a line counts, so her own voice can't
interrupt or command her.

Her own voice volume is a tool too: "speak up", "you're too loud", "talk at 50 percent"
(10-100%; more would only clip). It's saved, so it survives restarts.

The agent is a [Deep Agent](https://github.com/langchain-ai/deepagents) (a LangGraph
harness) in [agent/chat.py](agent/chat.py): the model either answers, or calls tools —
Spotify, its own voice volume, the PC's files and shell, or a sub-agent for a multi-step
job. Tool results (and errors) go back to the model, which then answers in a plain
sentence, so you hear "Playing Highway to Hell by AC/DC" rather than a raw API response.
The conversation is kept between turns and summarized before it outgrows Bonsai's
context. The current time rides on each user turn rather than the system prompt, so the
prompt prefix stays identical and llama-server reuses its cache.

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

Ask in plain language; the model maps it to a tool.

### Spotify

| | |
|---|---|
| **Playback** | play a track, playlist, or album; pause; resume; next; previous |
| **Queue** | add a track to the queue |
| **Volume** | set to a percentage; step up; step down |
| **Modes** | shuffle; repeat track; repeat playlist; repeat off |
| **Query** | what's playing right now |
| **Anything** | "open Spotify", "play some music", "surprise me" — a random pick, shuffled |

If the Spotify app isn't running, NIA opens it (through Windows' `spotify:` link — no
shell, nothing to approve) and waits for it to come online; if it's open but idle,
playback starts on it anyway, preferring this PC over a phone. When no song is named, she
picks at random from **Random music picks** in the control panel (`music_moods`, a
comma-separated list of playlist searches) and starts at a random track with shuffle on.
After every play she checks the music really started: a stuck Spotify app accepts
commands but loads nothing, and she'll say so rather than claim it's playing. When the
login expires ("Refresh token expired"), use **Spotify login** in the control panel.

### YouTube

| | |
|---|---|
| **Play** | "play X on YouTube" searches and opens the top result in your browser |
| **Transcript** | "summarize this video" reads what's said in the video last played (or a link, ID or search) |

No API key or account: search goes through [yt-dlp](https://github.com/yt-dlp/yt-dlp)
and transcripts through [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api).
Neither needs a spoken yes — the only thing they can open is a youtube.com watch link.
YouTube returns no results at all for some searches (age-restricted artists, for one);
NIA says so. Long transcripts are cut to the first few minutes to fit Bonsai's context.

### Her own voice

"Speak up", "you're too loud", "talk at 50 percent" set her voice volume (see above).

Adding a new capability means a `@tool` function the model can call — Spotify's are in
`_spotify_tools()` in [`agent/chat.py`](agent/chat.py), YouTube's in
[`agent/youtube.py`](agent/youtube.py). The tool's signature and docstring are what the
model sees, so the docstring is the prompt.

### Your PC

NIA can also work on the PC itself. Deep Agents' file tools only accept `/`-style
paths, so each drive is mounted as a folder — `C:\Users` is `/c/Users`, `D:\nia` is
`/d/nia` — while the shell takes normal Windows paths. Her prompt includes where
Windows really keeps your Desktop, Documents and Downloads (OneDrive often moves them).
She prefers the file tools, since reading needs no confirmation and the shell always does:

| | |
|---|---|
| **Read** — no confirmation | list folders, read files, find files by pattern, search inside files |
| **Change** — spoken yes first | write, edit and delete files; run `cmd.exe` commands (PowerShell via `powershell -Command`) |
| **Delegate** | hand a multi-step job to a sub-agent with its own context |

Before any change she says what she's about to do — *"Before I do that: I'll close the
YouTube tab. Should I go ahead?"* — in plain words, never the raw command (the log keeps
that). File changes name the file ("I'll save groceries.txt"); commands use her own short
account of what they do, and anything that deletes, closes programs, shuts down, formats or
touches the registry is always added out loud ("Note that it deletes files"), however
harmless the description sounds. Only a clear yes ("yes", "go ahead", "do it") runs it. Anything
else, including "yes… no wait", refuses it, and a question left unanswered for 60
seconds expires. **There is no sandbox:** the shell and files are the real PC, and that
spoken yes is the only guard — anyone within earshot can answer it.

## Tests

```bash
python test_agent.py
python test_voice.py
```

[test_voice.py](test_voice.py) checks the fuzzy wake-phrase matcher against real Moonshine
transcripts — what should wake her, what shouldn't, and where the threshold cuts — plus
the cleanup of replies before they're spoken and her voice volume.

[test_agent.py](test_agent.py) runs the real Deep Agent offline with a scripted model, a
fake Spotify client, a stubbed YouTube and an in-memory filesystem, so it needs no
network, no mic, no LLM server — and never touches the real PC. It checks that tool calls
reach the right place, that tool errors go back to the model instead of crashing, that PC
changes wait for a clear spoken yes (and a late or hedged one refuses), that long tool
results are cut and an overflowing conversation starts fresh, and that YouTube opens only
watch links without asking.

## Layout

```
main.py                             entry point
ui.py                               control panel: settings, start/stop everything
settings.py                         setting defaults (overrides in settings.json)
voice_assistant/voice_assistant.py  state machine, STT, TTS
run_bonsai.bat                      starts the LLM server (llama-server + Bonsai)
agent/chat.py                       Deep Agent, PC backend, approvals, Spotify tools
agent/action_controller.py          Spotify operations (and re-login: python -m agent.action_controller)
agent/youtube.py                    YouTube tools
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
- **Replies take ~2–4 s** from the end of your sentence: Bonsai reads the prompt and
  tools, calls a tool, then words the result. Kokoro adds ~1 s before speech starts; pick
  `piper_en_US-lessac-medium` as the voice in the control panel for ~0.2 s, sounding
  more robotic.
- **Spotify must be open somewhere.** An idle app is woken, but with no Spotify app
  running at all there's nothing to play on.

## Credits

NIA stands on these projects — thank you to everyone behind them. Each is under its own
license; see its page.

| Project | Used for |
|---|---|
| [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) by [PrismML](https://github.com/PrismML-Eng) | the LLM: a ternary build of [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-27B that fits in 8 GB of VRAM |
| [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), built on [llama.cpp](https://github.com/ggml-org/llama.cpp) | serving Bonsai (`llama-server`) with its ternary kernels |
| [Deep Agents](https://github.com/langchain-ai/deepagents) | the agent harness: tool loop, files and shell, sub-agents, approvals, summarization |
| [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain) (`langchain-openai`) | the graph Deep Agents runs on, tools, and the OpenAI-compatible client |
| [Moonshine](https://github.com/moonshine-ai/moonshine) (`moonshine-voice`) | wake phrase, speech recognition with voice-activity detection, and the TTS runtime |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and [Piper](https://github.com/rhasspy/piper) voices | NIA's voice, through Moonshine |
| [spotipy](https://github.com/spotipy-dev/spotipy) | Spotify Web API client |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | YouTube search and transcripts |
| [pycaw](https://github.com/AndreMiras/pycaw) | turning other apps down while NIA speaks (Windows mixer) |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | loading `.env` |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | the speaker embedding in `preprocess_voice.py` |

NIA isn't affiliated with or endorsed by any of these projects, Spotify or YouTube.

## License

MIT — see [LICENSE](LICENSE).
