"""Numbers as words, the way NIA says them - in her voice and in the window alike: "40%" -> "forty percent",
"1st" -> "first", "2026" -> "twenty twenty-six", "9:05" -> "nine oh five", "$6.30" -> "six dollars and thirty
cents", "v1.0.1" -> "one point oh point one". Her replies mixed "50 percent" with "fifty percent", and the voice
read digits unevenly. Links are left alone."""
import re

from num2words import num2words

URL = re.compile(r"\b(?:https?://|www\.)\S+")
# (one, many, one of the small unit, many of it); num2words' own currency mode said "five rupees" for 500
MONEY = {"$": ("dollar", "dollars", "cent", "cents"), "₹": ("rupee", "rupees", "paisa", "paise"),
         "£": ("pound", "pounds", "penny", "pence"), "€": ("euro", "euros", "cent", "cents")}
DIGITS = "oh one two three four five six seven eight nine".split()


def said(number, **how):
    """num2words without its commas, which the voice reads as pauses"""
    return num2words(number, **how).replace(",", "")


def _money(m):
    one, many, small, smalls = MONEY[m.group(1)]
    whole, cents = int(m.group(2).replace(",", "")), int((m.group(3) or "0").ljust(2, "0")[:2])
    words = f"{said(whole)} {one if whole == 1 else many}"
    return words + (f" and {said(cents)} {small if cents == 1 else smalls}" if cents else "")


def _time(m):
    hour, minute = int(m.group(1)), int(m.group(2))
    return f"{said(hour)} " + ("o'clock" if minute == 0 else f"oh {said(minute)}" if minute < 10 else said(minute))


def _number(m):
    text = m.group(0).replace(",", "")
    if len(text.lstrip("-")) >= 7 and "." not in text:  # a phone number or an id: digit by digit
        return " ".join(DIGITS[int(d)] if d != "0" else "zero" for d in text.lstrip("-"))
    if re.fullmatch(r"1[1-9]\d\d|20\d\d", m.group(0)):  # a year - "1,234" isn't one
        return said(int(text), to="year")
    return said(float(text) if "." in text else int(text))


def _version(m):
    spoken = " point ".join(DIGITS[0] if part == "0" else said(int(part)) for part in m.group(2).split("."))
    return ("version " if m.group(1) else "") + spoken


def _decade(m):
    """the 1990s -> nineteen nineties, the 80s -> eighties"""
    number = int(m.group(1))
    words = said(number, to="year") if number >= 1000 else said(number)
    return words[:-1] + "ies" if words.endswith("y") else words + "s"


WHOLE =r"(?:\d{1,3}(?:,\d{3})+|\d+)"  # 1,234 or 1234 - a comma counts only between groups of three
# In order: the special forms first, then whatever numbers are left
RULES = [(rf"([$₹£€])\s?({WHOLE})(?:\.(\d{{1,2}}))?\b", _money),
         (r"\b(\d{1,2}):(\d{2})\b", _time),
         (r"(?<![\w.])(v?)(\d+(?:\.\d+){2,})\b", _version),
         (r"\b(\d{1,3}0)s\b", _decade),
         (r"~\s?(?=\d)", "about "),
         (r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: said(int(m.group(1)), to="ordinal")),
         (r"(\d)\s?%", r"\1 percent"),
         (r"\s?°\s?C\b", " degrees"),
         (r"\s?°\s?F\b", " degrees Fahrenheit"),
         (r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " "),  # "27B" -> "27 B", "mp3" -> "mp 3"
         (rf"(?<![\w.])-?{WHOLE}(?:\.\d+)?(?!\d)", _number)]


def _words(text):
    for pattern, spoken in RULES:
        text = re.sub(pattern, spoken, text)
    return text


def in_words(text):
    """text with every number written as words, links left as they are"""
    parts = URL.split(text)
    links = URL.findall(text)
    return "".join(_words(part) + (links[i] if i < len(links) else "") for i, part in enumerate(parts))
