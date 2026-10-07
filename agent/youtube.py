"""YouTube for NIA: search, play a video, and read transcripts. No API key or account.

None of the tools need approval: search_youtube and youtube_transcript only read, and
play_youtube can only open a youtube.com watch link - at most one every OPEN_COOLDOWN seconds,
so a model that keeps "trying" can't fill the screen with tabs.
"""
import re
import time
import webbrowser

from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# An 11-character video ID, bare or inside a watch / youtu.be / shorts URL
VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|^)([\w-]{11})(?![\w-])")
OPEN_COOLDOWN = 15  # seconds between videos opened

last_video = None  # (id, title) of the video last played, so "summarize this video" works
last_opened = 0.0  # when play_youtube last opened a tab


def search(query, count=1):
    """[(id, title, channel)] of YouTube's top results for query - possibly none: YouTube returns
    nothing at all for some queries (e.g. age-restricted artists)"""
    with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
        entries = ydl.extract_info(f"ytsearch{count}:{query}", download=False).get("entries") or []
    return [(v["id"], v.get("title"), v.get("channel") or v.get("uploader")) for v in entries if v]


@tool
def search_youtube(query: str) -> str:
    """List YouTube's top 5 results for query WITHOUT opening anything - to look something up,
    check a name, or choose between videos before playing one. Use the user's exact words
    as the query; don't change or "correct" names"""
    results = search(query, 5)
    if not results:
        return f"YouTube returned no results for '{query}'"
    return "\n".join(f"{i}. {title} - {channel} (id {video_id})" for i, (video_id, title, channel) in
                     enumerate(results, 1))


@tool
def play_youtube(query: str) -> str:
    """Open ONE video on YouTube in the browser. query is a video id or link (e.g. from
    search_youtube), or a search whose top result is opened - use the user's exact words and
    don't change or "correct" names. For "play X on YouTube". Call it once per request: if the
    result isn't what was wanted, use search_youtube and ask the user, don't open more videos.
    Not for music on Spotify"""
    global last_video, last_opened
    wait = OPEN_COOLDOWN - (time.time() - last_opened)
    if wait > 0:
        return (f"Not opened: a video was opened {OPEN_COOLDOWN - wait:.0f} seconds ago. Don't open "
                "another - tell the user what's open and ask before trying again")
    if match := VIDEO_ID.search(query.strip()):
        video_id, title, channel = match.group(1), "the requested video", "YouTube"
    else:
        results = search(query)
        if not results:
            return f"YouTube returned no results for '{query}'"
        video_id, title, channel = results[0]
    webbrowser.open(f"https://www.youtube.com/watch?v={video_id}")
    last_video, last_opened = (video_id, title), time.time()
    return f"Opened '{title}' by {channel} on YouTube"


@tool
def youtube_transcript(video: str = "") -> str:
    """What is said in a YouTube video, to summarize it or answer questions about it.
    video is a YouTube link, a video ID, or a search; leave it empty for the video last played"""
    if not video:
        if not last_video:
            return "No video has been played yet - say which one"
        video_id, title = last_video
    elif match := VIDEO_ID.search(video.strip()):
        video_id, title = match.group(1), video
    else:
        results = search(video)
        if not results:
            return f"YouTube returned no results for '{video}'"
        video_id, title, _ = results[0]
    # ponytail: long videos get cut to the first few minutes by the agent's tool-result cap;
    # summarize in chunks if whole-video summaries matter
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US", "en-GB", "hi"))
    return f"Transcript of '{title}':\n" + " ".join(snippet.text for snippet in transcript)


TOOLS = [search_youtube, play_youtube, youtube_transcript]
