"""System 1: the commands NIA knows by heart, run at once without the language model.

"Pause the music" took 3-4 s through Bonsai - reading a 6K-token prompt, choosing pause, then writing a reply nobody
heard (minor actions go unspoken). The Spotify call itself is a fraction of that. A command here runs its tool
directly; anything else - a phrasing not listed, a second request, a "don't", a "later" - goes to Bonsai (System 2),
as does any reflex that fails, so Bonsai can deal with the error.

Strict on purpose: the whole utterance must be the command, give or take politeness and her name. A miss costs only
the usual 3-4 s; a wrong match would do the wrong thing.
"""
import re

# Said around a command without changing it: "Okay Nia, could you pause the music please, bro". Never "and" (more
# was coming) or "go" ("go to the next song")
POLITE = r"(?:can you|could you|would you|will you|thank you|for me|tell me|by the way|ok(?:ay)?|hey|hi|so|alright|" \
         r"right|nia|nea|niya|please|just|now|sir|bro|bra|man|thanks|though)"
EDGES = re.compile(rf"^(?:{POLITE}\s+)+|(?:\s+{POLITE})+$")
SPOTIFY = re.compile(r"\b(?:(?:on|in|for|of)\s+)?spotify\b")  # where the music is - it changes nothing here
YOUR = re.compile(r"\b(?:your|you're|youre|yourself|you\s+are)\b")
LEVELS = {"full": 100, "max": 100, "maximum": 100, "half": 50}
NUMBER = r"(?P<n>\d{1,3}|full|max|maximum|half)"
MUSIC = r"(?:the\s+|my\s+|this\s+)?(?:music|song|track|playback|it|spotify)"
SONG = r"(?:song|track|one)"

# (pattern, tool, arguments) - the pattern must match the whole cleaned-up utterance
MUSIC_COMMANDS = [
    (rf"pause(?:\s+{MUSIC})?|stop\s+{MUSIC}", "pause", {}),
    (rf"(?:resume|unpause)(?:\s+{MUSIC})?|(?:continue|keep)\s+playing(?:\s+{MUSIC})?|continue\s+{MUSIC}", "resume", {}),
    (rf"next(?:\s+{SONG})?|skip(?:\s+(?:this|the)?\s*{SONG})?|skip\s+it|(?:go\s+to|play|change\s+(?:it\s+)?to)\s+"
     rf"(?:the\s+)?next(?:\s+{SONG})?|change\s+(?:the\s+)?{SONG}", "skip", {"direction": "next"}),
    (rf"previous(?:\s+{SONG})?|(?:go\s+)?back\s+(?:a|one)\s+{SONG}|(?:go\s+(?:back\s+)?to|play)\s+the\s+previous"
     rf"(?:\s+{SONG})?", "skip", {"direction": "previous"}),
    (r"what(?:'s|\s+is)\s+(?:playing|this\s+song)|what\s+song\s+is\s+(?:this|playing)|what\s+am\s+i\s+listening\s+to|"
     r"who\s+(?:sings|is\s+singing)\s+this", "now_playing", {}),
]
# Volume: Spotify's, or hers when it's "your volume" / "speak up"
VOLUME = r"(?:the\s+)?(?:music\s+)?volume"
SET = rf"(?:(?:set|put|turn|change|make|bring|increase|raise|decrease|lower|reduce)(?:\s+(?:up|down))?\s+)?" \
      rf"{VOLUME}\s+(?:(?:to|at)\s+)?{NUMBER}(?:\s*(?:%|percent))?"
UP = rf"(?:turn|crank)\s+(?:up\s+{VOLUME}|{VOLUME}\s+up)|(?:increase|raise)\s+{VOLUME}|{VOLUME}\s+up"
DOWN = rf"turn\s+(?:down\s+{VOLUME}|{VOLUME}\s+down)|(?:decrease|lower|reduce)\s+{VOLUME}|{VOLUME}\s+down"
VOICE = r"(?:your\s+)?(?:voice\s+)?volume|your\s+voice"
VOICE_SET = rf"(?:(?:set|put|turn|change|make|increase|decrease)\s+)?(?:{VOICE})\s+(?:to|at)\s+{NUMBER}" \
            r"(?:\s*(?:%|percent))?"
VOICE_UP = rf"(?:speak|talk)\s+(?:up|louder)|(?:turn\s+up|increase|raise)\s+(?:{VOICE})|(?:{VOICE})\s+up|" \
           rf"turn\s+(?:{VOICE})\s+up|be\s+louder"
VOICE_DOWN = rf"(?:speak|talk)\s+(?:quieter|softer|more\s+quietly|lower)|(?:turn\s+down|decrease|lower|reduce)\s+" \
             rf"(?:{VOICE})|turn\s+(?:{VOICE})\s+down|(?:you're|youre|you\s+are)\s+too\s+loud|be\s+quieter"


def clean(text):
    """The words of an utterance, lower case, without the politeness around them"""
    text = re.sub(r"[^\w%'\s]", " ", text.lower().replace("’", "'"))
    text = " ".join(SPOTIFY.sub(" ", text).split())
    while (shorter := EDGES.sub("", text)) != text:
        text = shorter
    return text


def level(match):
    n = match.group("n")
    return LEVELS.get(n) or int(n)


def match(utterance):
    """(tool, arguments) for a command NIA knows by heart, or None - for Bonsai to work out"""
    said = utterance.lower()
    text = clean(said)
    if not text:
        return None
    hers = bool(YOUR.search(said)) and not SPOTIFY.search(said)  # "your volume" - unless Spotify is named
    if hers:
        if m := re.fullmatch(VOICE_SET, text):
            return ("set_voice_volume", {"percent": level(m)}) if 10 <= level(m) <= 100 else None
        if re.fullmatch(VOICE_UP, text):
            return "change_voice_volume", {"step": 25}
        if re.fullmatch(VOICE_DOWN, text):
            return "change_voice_volume", {"step": -25}
        return None  # anything else about her goes to Bonsai, never to Spotify
    if re.fullmatch(VOICE_UP, text) and "voice" not in text and "volume" not in text:  # "speak up", "talk louder"
        return "change_voice_volume", {"step": 25}
    if re.fullmatch(VOICE_DOWN, text) and "voice" not in text and "volume" not in text:
        return "change_voice_volume", {"step": -25}
    if "voice" in text:
        return None
    if m := re.fullmatch(SET, text):
        return ("set_volume", {"percent": level(m)}) if 0 <= level(m) <= 100 else None
    if re.fullmatch(UP, text):
        return "change_volume", {"step": 10}
    if re.fullmatch(DOWN, text):
        return "change_volume", {"step": -10}
    for pattern, tool, args in MUSIC_COMMANDS:
        if re.fullmatch(pattern, text):
            return tool, dict(args)
    return None
