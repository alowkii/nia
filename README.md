<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/svg/nia-logo-tagline-white.svg">
    <img src="assets/logo/svg/nia-logo-tagline-ink.svg" alt="NIA — Next-gen Intelligence Agent" width="440">
  </picture>
</p>

# NIA — Next-gen Intelligence Agent

A Jarvis-style voice assistant that runs entirely on your own machine. Wake phrase,
speech recognition, reasoning, and speech synthesis are all local — nothing is sent
to a cloud AI API. The only network calls are to Spotify, YouTube and web search (DuckDuckGo, and the
pages it finds), and only when you ask for them, plus the window's two fonts from Google Fonts (it falls back to Windows'
own fonts offline); the only account needed is a Spotify developer app.

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
threshold (default 0.8, in the settings panel) is the knob: lower wakes more easily but
false-triggers more — "Hey Mia" already scores 0.83. The mic is muted while NIA thinks
and speaks, so she doesn't transcribe herself. While she speaks, every other app (Spotify,
browser, games) is turned down to 30% of its volume in the Windows mixer and restored
afterwards — "Other apps' volume while NIA speaks" in the settings panel; 1 turns it off.
**Interrupting:** while she's thinking or talking, say **"Stop"**, **"Nia, stop"** or
**"Hey Nia"**. Her voice cuts off within a fraction of a second (up to ~1 s if she was
still preparing the first sentence — that synthesis is never aborted, since aborting it
crashed the process), and an unfinished turn
ends before its next step (steps already taken stay taken — an opened tab stays open).
"Hey Nia, pause the music" both stops her and does it. The mic stays live while she
works, but only a stop phrase at the start of a line counts, so her own voice can't
interrupt or command her.

**Short answers:** replies are spoken as plain sentences (any markdown is stripped) and cut
after about 40 words at a sentence end — she then asks *"Shall I go on, sir?"* and a "yes",
"go on" or "tell me more" reads the rest ("Max words spoken" in the settings panel). Her
name alone ("Nia", or "Near" as Moonshine often hears it) gets the greeting, not a guess at
a song called Near, and a clipped scrap like "Jo." gets *"Sorry sir, I only caught 'Jo'"*
instead of an action.

**Manner:** JARVIS from Iron Man — formal, unhurried and dry: *"Very good, sir. Thunderstruck is
playing now."*, *"I'm afraid Spotify isn't responding, sir. Shall I restart it?"* No exclamation
marks or "Enjoy!", and no wit when something has failed. Waking her gets a different greeting
each time, picked from **Greetings** in the settings panel (`|` between them) plus a few for the
hour: *"Good morning, sir."* at 8, *"Burning the midnight oil, sir?"* at 2 am. The voice is Kokoro's British
`bm_george`; any American or British Kokoro or Piper voice can be picked in the settings panel
(`af_heart` was the original). It runs at 1.2× speed ("NIA's speaking speed" in the panel).

**Rebooting:** "reboot" or "restart yourself" — she says she'll be back in a moment, then all of NIA restarts:
`nia.py`, the window's page, the language model, speech and agent, so any change to her code takes effect. The
window stays open and reloads itself; ~40 s later she's back at *Standing by* with the chime, the conversation
fresh and memory kept. If something of hers is stuck (she keeps mishearing), she may restart just her speech and
agent on her own (~10 s). Ollama is left running; its models hold no NIA code.

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
  in the settings panel if yours live elsewhere
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

One script runs everything, in one window:

```bash
python nia.py
```

(`pythonw nia.py`, or a shortcut to it, runs it without a console.) It opens NIA's window and, hidden
behind it, starts the LLM server if it isn't already up (~20 s) and the assistant. When the orb shows
*Standing by* — with a soft, warm two-note chime as she's ready (*Chime when she's ready* in the settings turns
it off) — say "Hey Nia", or type into the command line at the bottom. **Closing the window shuts everything
down**, including the LLM server if NIA started it. Running it again while NIA is up just opens another window
onto her. Say "reboot" to restart all of it — see **Rebooting** under [How it works](#how-it-works).

### The window

- **Centre:** the orb — rings, core and label in the colour of her state: *Booting*, *Standing by*
  (waiting for the wake phrase), *Listening* (it ripples, and shows your words as they're heard),
  *Thinking*, speaking (the core and the bars move with the loudness of each word), and *Offline*.
- **Left:** the voice pipeline — which stage is working, with the actual tool in use while she
  thinks (Spotify, YouTube, shell, files, memory…) and the LLM server's state.
- **Right:** the conversation — what you said and her reply, shown even when spoken replies are off.
- **Bottom:** her state, the mic button — tap to wake her or put her to sleep (or to restart her when
  offline); hold it in push-to-talk mode — the microphone she listens through (pick another from the
  menu beside the mic button; it switches at once), and a command line for typing instead of talking.
- **⚙ Settings:** the theme (six colour sets), the wake word / push-to-talk / spoken-replies switches,
  and every other setting: the LLM server, voice, listening, memory and approvals, music. Saving
  restarts only what needs it — voice speed or greetings apply at once; a new voice restarts the
  assistant (a few seconds); server settings restart the server too. Spotify login and an LLM server
  restart are there as well.

The window is [hud/hud.html](hud/hud.html) in an Edge app window, served by [nia.py](nia.py) on
`127.0.0.1:8765` and updated live over an event stream. The assistant runs as a hidden child process
(main.py) joined to it by an authenticated local connection ([utils/hud.py](utils/hud.py)); only that
window can drive it — a web page can't send its requests. Settings are saved to `settings.json`; the
defaults live in [settings.py](settings.py).

To run just the assistant, with its log in the console and no window: `python main.py`. For a
keyboard-only session with no audio at all — handy for iterating on prompts — `python test.py`.

## Logs

Everything lands in `logs/`:

| File | What's in it |
|---|---|
| `nia.log` | Each turn — what was heard or typed (and transcription latency), tool calls with arguments and results, the reply, LLM and speech timings — plus startup, settings changes, and every crash with its full traceback. Rotates at 5 MB, keeping 3 old files. |
| `assistant.out.log` | The hidden assistant's console output — mostly a copy of the above, plus anything a native library prints as it crashes. |
| `window.log` | The window's script (nia.py): starting and stopping the server and assistant, settings changes. |
| `test.log` | What the test suites log, kept out of NIA's real log. |
| `llama-server.log` | The LLM server's own output: model loading, per-request token speeds, errors. Rewritten each time the server starts. |

If she crashes, the window shows *Offline · stopped unexpectedly*; tap the mic to restart her.

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
| **Restart** | "restart Spotify" — force-closes and reopens only the Spotify app, no approval needed |

If the Spotify app isn't running, NIA opens it (through Windows' `spotify:` link — no
shell, nothing to approve) and waits for it to come online; if it's open but idle,
playback starts on it anyway, preferring this PC over a phone. When no song is named, she
picks at random from **Random music picks** in the settings panel (`music_moods`, a
comma-separated list of playlist searches), trying other picks if a search finds nothing, and
starts at a random track with shuffle on. A song is played inside its album, starting at that
track — Spotify's desktop app silently ignores a lone track sent on its own. If a freshly
opened app isn't ready yet (a 404), she retries once.
After every play she checks the music really started: a stuck Spotify app accepts
commands but loads nothing, and she'll say so rather than claim it's playing — offering a
restart, but never a second one within 10 minutes. When the
login expires ("Refresh token expired"), use **Spotify login** in the settings panel.

### YouTube

| | |
|---|---|
| **Play** | "play X on YouTube" searches and opens the top result in your browser |
| **Transcript** | "summarize this video" reads what's said in the video last played (or a link, ID or search) |
| **Websites** | "open Google", "show me the news" — any http(s) address, opened in your browser |

No API key or account: search goes through [yt-dlp](https://github.com/yt-dlp/yt-dlp)
and transcripts through [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api).
None of these needs a spoken yes — they can only open a web address in the browser, never
run anything, and at most one tab every 15 seconds.
YouTube returns no results at all for some searches (age-restricted artists, for one);
NIA says so. Long transcripts are cut to the first few minutes to fit Bonsai's context.

### The web, and jobs with several steps

| | |
|---|---|
| **Look it up** | "why does an octopus's heart stop when it swims?" — she searches, reads the best one to three pages and answers in her own words |
| **Several steps** | "check the weather in Mumbai tonight, play a playlist that suits it at 30 percent, and save a note about it" |

Search goes through DuckDuckGo's lite page and pages are read with the standard library
([agent/web.py](agent/web.py)) — no API key; each page is trimmed to fit Bonsai's context. Neither tool needs a
yes: they only read. For a job of three or more steps she writes a plan (LangChain's `write_todos`) and ticks it
off as she goes; the window shows it beside the conversation ("PLAN · 2/4", the current step under the orb). A
job that changes something asks once before the change ("I'll save evening.txt. Should I go ahead?"), then carries
on. Measured on Bonsai: a look-up ~10 s, a four-step job with research ~40 s. Plans of three to six steps work
well; much longer ones outgrow a 1-bit model's 16K context. Shell commands that reach the internet (`curl`,
`Invoke-WebRequest`...) always ask first — the web tools are the way online.

### Ask Claude

When she isn't sure of something, or it could have changed — news, weather, scores, prices, anything recent — NIA
hands the question to [Claude Code](https://claude.com/claude-code), which always browses — searches, then opens and
reads the most recent relevant page — and answers from what it found today, with the date; she says
"One moment, sir" while it works and passes the answer on without changing names, dates or numbers. Settled facts
("what's the capital of France?") she still answers herself, instantly. Measured: ~15–30 s per question at low
effort (twice that at Claude Code's default). The log shows how many searches and pages each answer took.

It runs Claude Code headless with fixed, locked-down settings ([agent/claude.py](agent/claude.py)): web search and
page fetching only (`--restricted` removes every tool that runs commands), anything else refused without a prompt
(`--permission-mode dontAsk`), no MCP servers, no saved session, in an empty folder — so whoever talks to NIA can
get answers through it but can't make it touch the PC. A shell command that starts `claude` itself always asks
first. Each question uses the Claude plan Claude Code is logged in with (~$0.03–0.08 at list price); the model and
effort are in the settings panel. Without Claude Code installed, or if a call fails, she falls back to `web_search`.
Answers to "most recent…" questions can differ between runs as search results do — higher effort checks more
sources, slower.

### Changing herself, on a branch

"Ask Claude to integrate Discord into you": on a spoken yes, NIA has Claude Code build the change in a separate
checkout (`D:\nia-changes\`) on a new branch, `nia/<what>-<when>` — her running code and your working folder are
never touched. Claude may edit only inside that checkout, never the approval rules, the Claude lock-down, the window
script or the existing tests, and run nothing but the test suites. When it's done NIA runs every test herself: if all
pass, the change is committed to the branch and she tells you its name, for you to review and merge (`git diff
HEAD..nia/...`, then `git merge`); if any fail, or Claude changes nothing, the branch is discarded. One change at a
time, in the background — she stays usable meanwhile. Measured: a small new tool took ~1–2 minutes and $0.25–0.45
at list price, on your Claude plan.

One honest limit: running the tests runs the code Claude just wrote, on this PC, before you've reviewed it. The
protected files and the spoken yes keep this to changes you asked for — but if that's too much, the tests can be
skipped so nothing runs until you've read the branch.

### Memory

NIA remembers across restarts, and only what's relevant reaches the LLM ([agent/memory.py](agent/memory.py)):

- **Facts** — "remember that Priya's birthday is June 3" (or anything lasting you mention) is saved
  with the `remember` tool; "forget …" deletes it.
- **Skills** — after a multi-step job, "remember how to do that as my evening routine" saves its steps by name
  (`save_skill`; saving under the same name updates it). Later — in any conversation — "do my evening routine"
  recalls the skill and she follows its steps instead of working the job out again. "What routines do you know?"
  lists them; "forget my evening routine" deletes one. The idea comes from [memU](https://github.com/NevaMind-AI/memU),
  kept inside NIA's own memory: no extra model calls, no cloud.
- **Past exchanges** — every finished turn worth keeping is saved too (not "I guess so", or a sentence cut off at
  "..."), and a search returns one of each: the same question asked five times comes back once.
- **Recall by meaning** — each message is embedded with [EmbeddingGemma](https://ollama.com/library/embeddinggemma)
  (on the CPU through Ollama, ~25 ms) and the few most similar memories — at most 4, only above the
  match threshold — are attached to it. An off-topic message gets none.
- **A short working history** — the LLM sees only recent exchanges (4 to 8; it cuts in steps of 4 so
  llama-server's prompt cache survives most turns). Older ones come back through recall when they
  matter, so the conversation stays around 6K tokens instead of growing until it's summarized.

Memories live in `memory/nia_memory.sqlite` (kept out of git), searched by plain cosine similarity —
milliseconds into the thousands. Set it up once with `ollama pull embeddinggemma`; clear **Memory
embedding model** in the settings panel to turn memory off. EmbeddingGemma was picked over all-minilm,
granite-embedding and nomic-embed-text because it alone separated relevant memories (similarity ≥ 0.29)
from off-topic questions (≤ 0.16) on a test set.

Most of each request is fixed, though: NIA's instructions and ~30 tool definitions are ~5.3K tokens
every turn, cached by llama-server. Memory bounds the part that grows, not that.

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
| **Run commands** — see below | `cmd.exe` commands (PowerShell via `powershell -Command`) |
| **Change files** — spoken yes first | write, edit and delete files |
| **Delegate** | hand a multi-step job to a sub-agent with its own context |

Commands go through three checks in [agent/approval.py](agent/approval.py), cheapest first:

1. **Risk rule** (code) — anything that deletes, closes programs, shuts down, installs, formats,
   touches the registry, presses keys in another window or runs as administrator always asks.
2. **Read-only allowlist** (code, instant) — one known read command (`tasklist`, `dir`, `type`,
   `ipconfig`, `git status`, PowerShell `Get-*` …), piped only into filters like `findstr`, with
   nothing chained or redirected, just runs.
3. **Decision model** — everything else is scored by [OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne-Ollama)
   (0.8B, on the CPU through Ollama's `/v1/systemone`): six narrow questions — does it write files,
   delete, open or start something, stop programs, install, run code? If every one is under the
   threshold (0.5) it runs; otherwise she asks. ~4–5 s per check; if Ollama is down, she asks.

Set it up once:

```bash
ollama pull hf.co/iapp/OpenThai-SystemOne-Ollama:Q8_0
ollama create openthai-cpu -f openthai-cpu.Modelfile
```

The Modelfile adds the `decision` capability the upstream build is missing and keeps it off the GPU.
Clear **Approval model** in the settings panel to skip this layer (anything not on the allowlist
then asks). [eval_approval.py](eval_approval.py) re-runs the 40 commands the design was chosen on
(20 harmless, 20 risky, 16 of them unseen by the questions): currently 20/20 harmless run, 0/20
risky. Every decision and its scores are logged.

Small things don't ask at all: making a **new folder** or saving a **new file** in your own folders (Desktop,
Documents, Downloads… and other drives) just happens — "make a folder called Ideas on my desktop and save a note in
it" is done in seconds. Overwriting or editing an existing file, deleting, and anything in Windows, Program Files,
AppData (the Startup folder runs whatever lands there), hidden dot-folders or NIA's own code still asks.

Before any other change she says what she's about to do — *"Before I do that: I'll close the
YouTube tab. Should I go ahead?"* — in plain words, never the raw command (the log keeps
that). File changes name the file ("I'll save groceries.txt"); commands use her own short
account of what they do, and anything that deletes, closes programs, shuts down, formats or
touches the registry is always added out loud ("Note that it deletes files"), however
harmless the description sounds. Only a clear yes ("yes", "go ahead", "go on", "do it", "sounds good"…) runs it.
Anything else holds off: a no ("no", "yes… no wait", "not now") is a no; an answer that's neither ("what?") is
treated as "not yet", and she asks again rather than saying you declined. A question left unanswered for 60
seconds expires. A no holds for the rest of that turn: she doesn't try another way to do
it (no YouTube in place of a refused command) — every tool but the read-only ones is
blocked until you ask for something new. **There is no sandbox:** the shell and files are the real PC, and that
spoken yes is the only guard — anyone within earshot can answer it.

## Tests

```bash
python test_agent.py
python test_voice.py
python test_nia.py
```

[test_voice.py](test_voice.py) checks the fuzzy wake-phrase matcher against real Moonshine
transcripts — what should wake her, what shouldn't, and where the threshold cuts — plus
the cleanup of replies before they're spoken (markdown, the ~40-word cut), how her name
alone and clipped fragments are recognised, greetings, her voice volume, each sentence's loudness
curve, the ready chime (warm, never tinny, no click, played once she's ready), and the voice loop driven from
the window: typed commands, the mic button (wake, sleep, push-to-talk), settings that apply at once, asking to
restart or reboot, and quitting.

[test_nia.py](test_nia.py) checks the window's script: the page and its live state stream, typed
commands and mic presses reaching the assistant, settings validated, saved and applied (at once, or by
restarting only what needs it), a reboot she asks for handing over to a fresh nia.py, the logo files, and that
no other web page can drive NIA.

[test_agent.py](test_agent.py) runs the real Deep Agent offline with a scripted model, a
fake Spotify client, a stubbed YouTube and an in-memory filesystem, so it needs no
network, no mic, no LLM server — and never touches the real PC. It checks that tool calls
reach the right place, that tool errors go back to the model instead of crashing, that PC
changes wait for a clear spoken yes (and a late or hedged one refuses), that long tool
results are cut and an overflowing conversation starts fresh, that YouTube and websites open
only web addresses without asking, that a no blocks other actions that turn, and the Spotify
fixes from a real session (album playback, 404 retry, no repeat restarts).

## Layout

```
nia.py                              start here: NIA's window, with the LLM server and assistant behind it
hud/hud.html                        the window itself (HUD, settings panel)
assets/logo/                        the NIA logo: SVG and PNG in every lockup, app icons, favicon
main.py                             the assistant on its own (nia.py runs it hidden)
settings.py                         setting defaults and checks (overrides in settings.json)
voice_assistant/voice_assistant.py  state machine, STT, TTS
utils/hud.py                        the assistant's link to the window
agent/server.py                     the LLM server command and health check
run_bonsai.bat                      starts just the LLM server (llama-server + Bonsai)
agent/chat.py                       Deep Agent, PC backend, approvals, Spotify tools
agent/action_controller.py          Spotify operations (and re-login: python -m agent.action_controller)
agent/youtube.py                    YouTube tools
agent/memory.py                     long-term memory: facts and past exchanges, recalled by vector search
agent/prompts/                      system prompt
preprocess_voice.py                 builds a speaker embedding from a voice sample
test_agent.py, test_voice.py, test_nia.py  offline tests
eval_approval.py                    live check of the approval layers (needs Ollama)
agent/approval.py                   which PC commands run without asking
openthai-cpu.Modelfile              builds the approval model for Ollama
test.py                             text-only REPL
```

## Known limitations

- **Speaker verification is disabled.** The code to check that a command came from
  your voice is present but commented out in `voice_assistant.py` — it added too much
  latency per command. `preprocess_voice.py` still generates the reference embedding.
  Anyone within earshot can currently issue commands.
- **Replies take ~2–4 s** from the end of your sentence: Bonsai reads the prompt and
  tools, calls a tool, then words the result. Kokoro adds ~1 s before speech starts; pick
  `piper_en_US-lessac-medium` as the voice in the settings panel for ~0.2 s, sounding
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
| [EmbeddingGemma](https://ollama.com/library/embeddinggemma) by Google | embeddings for NIA's memory search |
| [OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne-Ollama) by iApp Technology, run by [Ollama](https://github.com/ollama/ollama) | the decision model that clears harmless PC commands — chosen with the [S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard) |
| [Rajdhani](https://fonts.google.com/specimen/Rajdhani) and [JetBrains Mono](https://www.jetbrains.com/lp/mono/) | the window's type |
| [pycaw](https://github.com/AndreMiras/pycaw) | turning other apps down while NIA speaks (Windows mixer) |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | loading `.env` |
| [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | the speaker embedding in `preprocess_voice.py` |

NIA isn't affiliated with or endorsed by any of these projects, Spotify or YouTube.

## License

MIT — see [LICENSE](LICENSE).
