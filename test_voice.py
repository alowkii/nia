"""Checks the voice text helpers: the fuzzy wake-phrase matcher on real Moonshine transcripts,
and the cleanup of replies before they're spoken. Run: python test_voice.py"""
from voice_assistant.voice_assistant import speakable, wake_command


def wake(text, threshold=0.8):
    return wake_command(text, "hey nia", threshold)


# Woken, with whatever follows the phrase as the command
assert wake("Hey Nia.") == ""
assert wake(" Hey Nia.") == ""
assert wake("Hey, Nia!") == ""
assert wake("Hey Nia play some loafy beats.") == "play some loafy beats."
assert wake("Hey Nia, what time is it?") == "what time is it?"
assert wake("Yeah. Hey Nia what time is it?") == "what time is it?"  # stray leading word
# Mid-line, as when music or talk runs into the phrase
assert wake("I just can't stop loving you Hey Nia, pause the music.") == "pause the music."

# Not woken
assert wake("Hey nice to meet you.") is None  # 0.77, just under the default
assert wake("Here is the news.") is None
assert wake("Hey, man.") is None
assert wake("") is None
assert wake("Tell me about California.") is None  # 'nia' inside a word

# The threshold is the knob: "Hey Mia" scores 0.83
assert wake("Hey Mia, how are you?") == "how are you?"
assert wake("Hey Mia, how are you?", threshold=0.9) is None

# Spoken text: no symbol may reach the voice, which reads non-ASCII as the letter "L"
assert speakable("I hit a hiccup, sir — the token expired.") == "I hit a hiccup, sir, the token expired."
assert speakable("Done, sir — volume’s at 50% now…") == "Done, sir, volume's at 50 percent now..."
assert speakable("Playing “Halo” by Beyoncé 🎵") == 'Playing "Halo" by Beyonce'
assert speakable("Some R&B, then AC/DC → enjoy!") == "Some R and B, then AC DC enjoy!"
assert speakable("Plain text stays as it is.") == "Plain text stays as it is."
assert speakable("Wait—what?") == "Wait, what?"
assert all(ord(c) < 128 for c in speakable("Café ☕ — naïve “quotes” ‘and’ © 2026 ✓"))

print("ok")
