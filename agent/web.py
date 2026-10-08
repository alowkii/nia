"""Web search and page reading for NIA, so research doesn't go through the shell. No API key, standard library only.

Search is DuckDuckGo's lite page - its main HTML page answers scripted requests with a bot check. Both tools only
read, so neither needs approval; results are trimmed to fit Bonsai's context.
"""
import html
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from langchain_core.tools import tool

AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"
PAGE_CHARS = 2800  # what one page may add to the conversation (tool results are capped at 3000)
LINK = re.compile(r"<a[^>]+href=\"([^\"]+)\"[^>]*class='result-link'>(.*?)</a>", re.S)
SNIPPET = re.compile(r"<td class='result-snippet'>(.*?)</td>", re.S)


def _get(url, timeout=10, limit=2_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept-Language": "en"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        kind = response.headers.get_content_type()
        charset = response.headers.get_content_charset() or "utf-8"
        return kind, response.read(limit).decode(charset, "replace")


def _text(fragment):
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def search(query, count=5):
    """[(title, url, snippet)] of the top results"""
    _, page = _get("https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query))
    results = []
    for (href, title), snippet in zip(LINK.findall(page), SNIPPET.findall(page) + [""] * 50):
        target = urllib.parse.parse_qs(urllib.parse.urlsplit(html.unescape(href)).query).get("uddg", [href])[0]
        if "duckduckgo.com/y.js" in target:  # an ad
            continue
        results.append((_text(title), target, _text(snippet)))
        if len(results) == count:
            break
    return results


class _Reader(HTMLParser):
    """A page's readable text: no scripts, styles, menus or footers; a line break at each block"""
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside", "button"}
    BLOCKS = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "section", "article", "blockquote"}

    def __init__(self):
        super().__init__()
        self.skipping, self.parts, self.title, self.in_title = 0, [], "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        self.in_title = tag == "title"

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skipping:
            self.parts.append(data)


def readable(page):
    """(title, text) of an HTML page - the lines that read like prose, not the link lists around them"""
    reader = _Reader()
    reader.feed(page)
    lines = (" ".join(line.split()) for line in "".join(reader.parts).splitlines())
    prose = [line for line in lines if len(line) >= 60 or (len(line) >= 25 and line[-1:] in ".!?:")]
    return " ".join(reader.title.split()), "\n".join(prose)


@tool
def web_search(query: str) -> str:
    """Search the web: the top 5 results, each with its title, address and a snippet. For facts, news, prices,
    how-tos, anything to look up or research. Then use read_page on the most promising results for detail"""
    try:
        results = search(query)
    except OSError as e:
        return f"The search didn't go through ({e}) - the internet may be down"
    if not results:
        return f"No results for '{query}'"
    return "\n".join(f"{i}. {title} - {url}\n   {snippet}" for i, (title, url, snippet) in enumerate(results, 1))


@tool
def read_page(url: str) -> str:
    """The main text of a web page (an http:// or https:// address, e.g. from web_search), trimmed to its first few
    paragraphs. Reads only - nothing opens on screen; use open_link to show a page to the user"""
    if urllib.parse.urlsplit(url.strip()).scheme not in ("http", "https"):
        return "Only http:// or https:// pages can be read"
    try:
        kind, page = _get(url.strip())
    except OSError as e:
        return f"Couldn't read that page ({e}) - try another result"
    if kind == "text/plain":
        return page[:PAGE_CHARS]
    if kind != "text/html":
        return f"That's a {kind} file, not a page I can read"
    title, text = readable(page)
    if not text:
        return f"'{title or url}' has no readable text (it may need JavaScript) - try another result"
    return f"{title}\n\n{text[:PAGE_CHARS]}" + (" [...]" if len(text) > PAGE_CHARS else "")


TOOLS = [web_search, read_page]
