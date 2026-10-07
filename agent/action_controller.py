import os
import time
from pathlib import Path
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

# Absolute, so the saved login is found whatever the working directory
CACHE = Path(__file__).resolve().parent.parent / ".spotify_cache"


def _describe(item):
    """'<name> by <artists>' for a track or album"""
    return f"{item['name']} by {', '.join(artist['name'] for artist in item['artists'])}"


class SpotifyController:
    """Spotify API errors propagate; the agent's ToolNode hands them back to the model."""

    def __init__(self):
        """Initialize Spotify client with authentication"""
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_ID"),
            client_secret=os.getenv("SPOTIFY_SECRET"),
            redirect_uri="https://aalokpandit.netlify.app/",
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
            cache_path=str(CACHE),
            open_browser=True
        ))

    def _start(self, **kwargs):
        """start_playback, waking an idle device when none is active - an open Spotify app
        that hasn't played recently is listed but idle, and Spotify won't pick it on its own"""
        devices = self.sp.devices()["devices"]
        if devices and not any(d["is_active"] for d in devices):
            # ponytail: takes the first listed device; choose by name if you run several
            kwargs["device_id"] = devices[0]["id"]
        self.sp.start_playback(**kwargs)

    def _search(self, query, kind):
        """Top hit for kind 'track', 'playlist' or 'album', or None"""
        # Spotify returns null entries for playlists it won't expose to apps, so the top one may be blank
        items = self.sp.search(q=query, limit=5, type=kind)[kind + "s"]["items"]
        return next((item for item in items if item), None)

    def get_current_playback(self):
        """Full playback state, or None if no active device"""
        return self.sp.current_playback()

    def get_current_track(self):
        """'<track> by <artists>', or None if nothing is loaded"""
        playback = self.get_current_playback()
        if not playback or not playback.get("item"):
            return None
        return _describe(playback["item"])

    def now_playing(self):
        """Answer for 'what's playing?'"""
        playback = self.get_current_playback()
        if not playback or not playback.get("item"):
            return "Nothing is playing right now"
        state = "playing" if playback.get("is_playing") else "paused"
        return f"{_describe(playback['item'])}, currently {state}"

    def play_track(self, track_name):
        """Search and play a track"""
        track = self._search(track_name, "track")
        if not track:
            return f"Track '{track_name}' not found"
        self._start(uris=[track['uri']])
        return f"Successfully playing: {_describe(track)}"

    def play_playlist(self, playlist_name):
        """Search and play a playlist"""
        playlist = self._search(playlist_name, "playlist")
        if not playlist:
            return f"Playlist '{playlist_name}' not found"
        self._start(context_uri=playlist['uri'])
        return f"Successfully playing playlist: {playlist['name']}"

    def play_album(self, album_name):
        """Search and play an album"""
        album = self._search(album_name, "album")
        if not album:
            return f"Album '{album_name}' not found"
        self._start(context_uri=album['uri'])
        return f"Successfully playing album: {_describe(album)}"

    def pause(self):
        """Pause playback"""
        self.sp.pause_playback()
        return "Playback paused successfully"

    def resume(self):
        """Resume playback"""
        self._start()
        return "Playback resumed successfully"

    def _after_skip(self, message):
        time.sleep(0.5)  # let Spotify update before we ask what's playing
        current = self.get_current_track()
        return f"{message}: {current}" if current else message

    def next(self):
        """Skip to next track"""
        self.sp.next_track()
        return self._after_skip("Skipped to next track")

    def previous(self):
        """Go to previous track"""
        self.sp.previous_track()
        return self._after_skip("Went back to previous track")

    def set_volume(self, volume_percent):
        """Set volume (0-100)"""
        try:
            volume_percent = int(volume_percent)
        except (ValueError, TypeError):
            return f"Invalid volume value: {volume_percent}. Must be a number between 0-100"
        if not 0 <= volume_percent <= 100:
            return "Volume must be between 0 and 100"
        self.sp.volume(volume_percent)
        return f"Volume set to {volume_percent}%"

    def change_volume(self, step):
        """Nudge volume by step percent (negative to lower), clamped to 0-100"""
        current = self.get_current_playback()
        if not current or not current.get('device'):
            return "No active device found to change volume"
        current_volume = current['device']['volume_percent']
        new_volume = max(0, min(current_volume + step, 100))
        self.sp.volume(new_volume)
        direction = "increased" if step > 0 else "decreased"
        return f"Volume {direction} from {current_volume}% to {new_volume}%"

    def shuffle(self, state=True):
        """Toggle shuffle mode"""
        self.sp.shuffle(state)
        return f"Shuffle {'enabled' if state else 'disabled'} successfully"

    def repeat(self, state='context'):
        """
        Set repeat mode
        state: 'track', 'context', or 'off'
        - 'track': repeat current track
        - 'context': repeat current context (playlist/album)
        - 'off': turn off repeat
        """
        if state not in ('track', 'context', 'off'):
            return "Invalid repeat state. Use 'track', 'context', or 'off'"
        self.sp.repeat(state)
        if state == 'track':
            return "Repeat mode set to: repeat current track"
        if state == 'context':
            return "Repeat mode set to: repeat playlist/album"
        return "Repeat mode turned off"

    def add_to_queue(self, track_name):
        """Add a track to the queue"""
        track = self._search(track_name, "track")
        if not track:
            return f"Track '{track_name}' not found"
        self.sp.add_to_queue(track['uri'])
        return f"Successfully added to queue: {_describe(track)}"


if __name__ == "__main__":
    # Log in to Spotify again, e.g. after "Refresh token expired": python -m agent.action_controller
    CACHE.unlink(missing_ok=True)
    print("Approve NIA in the browser tab that opens, then paste the address it redirects you to here.")
    print("Logged in. Spotify says:", SpotifyController().now_playing())
    input("Press Enter to close.")
