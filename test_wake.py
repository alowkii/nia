"""Checks the fuzzy wake-phrase matcher on real Moonshine transcripts. Run: python test_wake.py"""
from voice_assistant.voice_assistant import wake_command


def wake(text, threshold=0.8):
    return wake_command(text, "hey nia", threshold)


# Woken, with whatever follows the phrase as the command
assert wake("Hey Nia.") == ""
assert wake(" Hey Nia.") == ""
assert wake("Hey, Nia!") == ""
assert wake("Hey Nia play some loafy beats.") == "play some loafy beats."
assert wake("Hey Nia, what time is it?") == "what time is it?"
assert wake("Yeah. Hey Nia what time is it?") == "what time is it?"  # stray leading word

# Not woken
assert wake("Hey nice to meet you.") is None  # 0.77, just under the default
assert wake("Here is the news.") is None
assert wake("Hey, man.") is None
assert wake("") is None
assert wake("Tell me about California.") is None  # 'nia' inside a word

# The threshold is the knob: "Hey Mia" scores 0.83
assert wake("Hey Mia, how are you?") == "how are you?"
assert wake("Hey Mia, how are you?", threshold=0.9) is None

print("ok")
