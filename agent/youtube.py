"""YouTube for NIA: play the top search result, and read transcripts. No API key or account.

Neither tool needs approval: play_youtube can only open a youtube.com watch link, and
youtube_transcript only reads.
"""
import re
import webbrowser

from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

# An 11-character video ID, bare or inside a watch / youtu.be / shorts URL
VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|^)([\w-]{11})(?![\w-])")

last_video = None  # (id, title) of the video last played, so "summarize this video" works


def first_video(query):
    """(id, title, channel) of YouTube's top result for query, or None. YouTube returns
    nothing at all for some queries (e.g. age-restricted artists)"""
    with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
        entries = ydl.extract_info(f"ytsearch1:{query}", download=False).get("entries") or []
    if not entries:
        return None
    video = entries[0]
    return video["id"], video.get("title"), video.get("channel") or video.get("uploader")


@tool
def play_youtube(query: str) -> str:
    """Play a video on YouTube: searches for query and opens the top result in the browser.
    For "play X on YouTube", "show me a video of X". Not for music on Spotify"""
    global last_video
    found = first_video(query)
    if not found:
        return f"YouTube returned no results for '{query}'"
    video_id, title, channel = found
    webbrowser.open(f"https://www.youtube.com/watch?v={video_id}")
    last_video = (video_id, title)
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
        found = first_video(video)
        if not found:
            return f"YouTube returned no results for '{video}'"
        video_id, title, _ = found
    # ponytail: long videos get cut to the first few minutes by the agent's tool-result cap;
    # summarize in chunks if whole-video summaries matter
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US", "en-GB", "hi"))
    return f"Transcript of '{title}':\n" + " ".join(snippet.text for snippet in transcript)


TOOLS = [play_youtube, youtube_transcript]
