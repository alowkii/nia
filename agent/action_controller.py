import logging
import os
import random
import subprocess
import time
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import settings

logger = logging.getLogger(__name__)

# Absolute, so the saved login is found whatever the working directory
CACHE = Path(__file__).resolve().parent.parent / ".spotify_cache"


def _describe(item):
    """'<name> by <artists>' for a track or album"""
    return f"{item['name']} by {', '.join(artist['name'] for artist in item['artists'])}"


def oauth():
    """NIA's Spotify login; the token is saved to CACHE and refreshed from there"""
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_ID"),
        client_secret=os.getenv("SPOTIFY_SECRET"),
        redirect_uri="https://aalokpandit.netlify.app/",
        scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
        cache_path=str(CACHE),
        open_browser=True
    )


class SpotifyController:
    """Spotify API errors propagate; the agent's ToolNode hands them back to the model."""

    def __init__(self):
        """Initialize Spotify client with authentication"""
        self.sp = spotipy.Spotify(auth_manager=oauth())

    def open_app(self, timeout=20):
        """Launch Spotify on this PC unless it's already a device, and wait until it shows up as one.
        Uses Windows' spotify: link, so there's no shell command and nothing to approve."""
        devices = self.sp.devices()["devices"]
        if any(d["type"] == "Computer" for d in devices):
            return devices
        # Also wakes an app that's open but idle: Spotify drops those from the device list after a while
        os.startfile("spotify:")
        started = time.time()
        for _ in range(timeout * 2):
            time.sleep(0.5)
            devices = self.sp.devices()["devices"]
            if any(d["type"] == "Computer" for d in devices):
                logger.info(f"Spotify app came online after {time.time() - started:.1f}s")
                time.sleep(1)  # just registered: give the app a moment before it's sent a command
                return devices
        raise RuntimeError(f"Opened the Spotify app, but it didn't come online within {timeout} seconds")

    def restart_app(self):
        """Force-close Spotify on this PC and open it again - the fix for an app that accepts
        commands but plays nothing. Only ever touches Spotify's own processes"""
        for image in ("Spotify.exe", "SpotifyLauncher.exe"):
            subprocess.run(["taskkill", "/F", "/IM", image], capture_output=True)
        time.sleep(2)  # let Spotify drop off the device list before waiting for it to come back
        self.open_app()
        self.restarted = time.time()
        return "Restarted the Spotify app on this PC; it's back online"

    def _start(self, **kwargs):
        """start_playback, opening the Spotify app first if no device is available, and waking an
        idle device when none is active - an open Spotify app that hasn't played recently is listed
        but idle, and Spotify won't pick it on its own. Then checks something really started:
        Spotify accepts commands even when its app is stuck and loads nothing, and "Successfully
        playing" would be a lie."""
        started = time.time()
        devices = self.sp.devices()["devices"]
        logger.info(f"Spotify devices: {[(d.get('name'), d.get('type'), d.get('is_active')) for d in devices]}")
        if not any(d["is_active"] for d in devices) and not any(d["type"] == "Computer" for d in devices):
            devices = self.open_app()  # nothing playing anywhere and no app on this PC: open it here
        if devices and not any(d["is_active"] for d in devices):
            # Prefer a computer - NIA runs on one - over a phone that happens to be listed first
            device = min(devices, key=lambda d: d["type"] != "Computer")
            kwargs["device_id"] = device["id"]
        try:
            self.sp.start_playback(**kwargs)
        except spotipy.SpotifyException as e:
            if e.http_status != 404 or "device_id" not in kwargs:
                raise
            # Just after a restart the old device id can linger: look the device up again, once
            time.sleep(2)
            fresh = [d for d in self.sp.devices()["devices"] if d["type"] == "Computer"]
            kwargs["device_id"] = fresh[0]["id"] if fresh else kwargs.pop("device_id")
            self.sp.start_playback(**kwargs)
        for _ in range(6):  # ~3 s for the app to load the track
            time.sleep(0.5)
            playback = self.get_current_playback()
            if playback and playback.get("item") and playback.get("is_playing"):
                logger.info(f"Spotify playing after {time.time() - started:.1f}s")
                return
        where = playback["device"]["name"] if playback and playback.get("device") else "the device"
        if time.time() - getattr(self, "restarted", 0) < 600:
            # Restarting again won't help - one session tried it five times in a row
            raise RuntimeError(f"Spotify accepted the command but nothing started playing on {where}, even though "
                               "Spotify was restarted recently. Don't restart it again: tell the user it isn't playing")
        raise RuntimeError(f"Spotify accepted the command but nothing started playing on {where}. The app may be "
                           "stuck: tell the user, and offer to restart it with restart_spotify")

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
        # Inside its album, starting at the track: a lone track (uris=[...]) stopped starting on this PC's
        # Spotify app while albums and playlists kept working - measured: nothing in 4 s vs playing in 0.5 s
        self._start(context_uri=track["album"]["uri"], offset={"uri": track["uri"]})
        return f"Successfully playing: {_describe(track)}"

    def play_something(self, mood=""):
        """Play something when no song was named: a playlist for mood, or for a random pick from the
        music_moods setting, started at a random track with shuffle on"""
        # ponytail: picks from a fixed list in settings; replace with learned preferences later
        moods = [m.strip() for m in settings.load()["music_moods"].split(",") if m.strip()]
        random.shuffle(moods)
        playlist = None
        for pick in ([mood] if mood else []) + moods[:3]:  # a search can come back empty: try another mood
            if playlist := self._search(pick, "playlist"):
                break
        if not playlist:
            return "Found no playlist to play"
        total = (playlist.get("tracks") or {}).get("total") or 1
        self._start(context_uri=playlist["uri"], offset={"position": random.randrange(min(total, 100))})
        try:
            self.sp.shuffle(True)
        except spotipy.SpotifyException:  # 404 when Spotify hasn't caught up with the device yet - it's playing anyway
            return f"Playing the playlist {playlist['name']} (picked for '{pick}'), from a random track"
        return f"Playing the playlist {playlist['name']} (picked for '{pick}'), shuffled"

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
        """Set volume, clamped to 0-100"""
        volume_percent = max(0, min(int(volume_percent), 100))
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
        """Repeat 'track', 'context' (the playlist or album) or 'off'"""
        self.sp.repeat(state)
        return f"Repeat set to {state}"

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
