import os
import time
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

class SpotifyController:
    def __init__(self):
        """Initialize Spotify client with authentication"""
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv("SPOTIFY_ID"),
            client_secret=os.getenv("SPOTIFY_SECRET"),
            redirect_uri="https://aalokpandit.netlify.app/",
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
            cache_path=".spotify_cache",
            open_browser=True
        ))

    def get_current_playback(self):
        """Full playback state, or None if no active device"""
        return self.sp.current_playback()

    def get_current_track(self):
        """'<track> by <artists>', or None if nothing is loaded"""
        playback = self.get_current_playback()
        if not playback or not playback.get("item"):
            return None
        track = playback["item"]
        artists = ", ".join(artist["name"] for artist in track["artists"])
        return f"{track['name']} by {artists}"

    def now_playing(self):
        """Answer for 'what's playing?'"""
        try:
            playback = self.get_current_playback()
            if not playback or not playback.get("item"):
                return "Nothing is playing right now"
            track = playback["item"]
            artists = ", ".join(artist["name"] for artist in track["artists"])
            state = "playing" if playback.get("is_playing") else "paused"
            return f"{track['name']} by {artists}, currently {state}"
        except Exception as e:
            return f"Error getting current track: {e}"

    def play_track(self, track_name):
        """Search and play a track"""
        try:
            results = self.sp.search(q=track_name, limit=1, type='track')

            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                artists = ", ".join(artist['name'] for artist in track['artists'])
                self.sp.start_playback(uris=[track['uri']])
                return f"Successfully playing: {track['name']} by {artists}"
            return f"Track '{track_name}' not found"
        except Exception as e:
            return f"Error playing track: {e}"

    def play_playlist(self, playlist_name):
        """Search and play a playlist"""
        try:
            results = self.sp.search(q=playlist_name, limit=1, type='playlist')

            if results['playlists']['items']:
                playlist = results['playlists']['items'][0]
                self.sp.start_playback(context_uri=playlist['uri'])
                return f"Successfully playing playlist: {playlist['name']}"
            return f"Playlist '{playlist_name}' not found"
        except Exception as e:
            return f"Error playing playlist: {e}"

    def play_album(self, album_name):
        """Search and play an album"""
        try:
            results = self.sp.search(q=album_name, limit=1, type='album')

            if results['albums']['items']:
                album = results['albums']['items'][0]
                artists = ", ".join(artist['name'] for artist in album['artists'])
                self.sp.start_playback(context_uri=album['uri'])
                return f"Successfully playing album: {album['name']} by {artists}"
            return f"Album '{album_name}' not found"
        except Exception as e:
            return f"Error playing album: {e}"

    def pause(self):
        """Pause playback"""
        try:
            self.sp.pause_playback()
            return "Playback paused successfully"
        except Exception as e:
            return f"Error pausing playback: {e}"

    def resume(self):
        """Resume playback"""
        try:
            self.sp.start_playback()
            return "Playback resumed successfully"
        except Exception as e:
            return f"Error resuming playback: {e}"

    def next(self):
        """Skip to next track"""
        try:
            self.sp.next_track()
            time.sleep(0.5)  # let Spotify update before we ask what's playing
            current = self.get_current_track()
            return f"Skipped to next track: {current}" if current else "Skipped to next track"
        except Exception as e:
            return f"Error skipping track: {e}"

    def previous(self):
        """Go to previous track"""
        try:
            self.sp.previous_track()
            time.sleep(0.5)  # let Spotify update before we ask what's playing
            current = self.get_current_track()
            return f"Went back to previous track: {current}" if current else "Went back to previous track"
        except Exception as e:
            return f"Error going to previous track: {e}"

    def set_volume(self, volume_percent):
        """Set volume (0-100)"""
        try:
            volume_percent = int(volume_percent)
            if 0 <= volume_percent <= 100:
                self.sp.volume(volume_percent)
                return f"Volume set to {volume_percent}%"
            return "Volume must be between 0 and 100"
        except (ValueError, TypeError):
            return f"Invalid volume value: {volume_percent}. Must be a number between 0-100"
        except Exception as e:
            return f"Error setting volume: {e}"

    def increase_volume(self, step=10):
        """Increase volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current.get('device'):
                current_volume = current['device']['volume_percent']
                new_volume = min(current_volume + step, 100)
                self.sp.volume(new_volume)
                return f"Volume increased from {current_volume}% to {new_volume}%"
            return "No active device found to increase volume"
        except Exception as e:
            return f"Error increasing volume: {e}"

    def decrease_volume(self, step=10):
        """Decrease volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current.get('device'):
                current_volume = current['device']['volume_percent']
                new_volume = max(current_volume - step, 0)
                self.sp.volume(new_volume)
                return f"Volume decreased from {current_volume}% to {new_volume}%"
            return "No active device found to decrease volume"
        except Exception as e:
            return f"Error decreasing volume: {e}"

    def shuffle(self, state=True):
        """Toggle shuffle mode"""
        try:
            self.sp.shuffle(state)
            return f"Shuffle {'enabled' if state else 'disabled'} successfully"
        except Exception as e:
            return f"Error toggling shuffle: {e}"

    def repeat(self, state='context'):
        """
        Set repeat mode
        state: 'track', 'context', or 'off'
        - 'track': repeat current track
        - 'context': repeat current context (playlist/album)
        - 'off': turn off repeat
        """
        try:
            if state not in ('track', 'context', 'off'):
                return "Invalid repeat state. Use 'track', 'context', or 'off'"
            self.sp.repeat(state)
            if state == 'track':
                return "Repeat mode set to: repeat current track"
            if state == 'context':
                return "Repeat mode set to: repeat playlist/album"
            return "Repeat mode turned off"
        except Exception as e:
            return f"Error setting repeat: {e}"

    def add_to_queue(self, track_name):
        """Add a track to the queue"""
        try:
            results = self.sp.search(q=track_name, limit=1, type='track')

            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                artists = ", ".join(artist['name'] for artist in track['artists'])
                self.sp.add_to_queue(track['uri'])
                return f"Successfully added to queue: {track['name']} by {artists}"
            return f"Track '{track_name}' not found"
        except Exception as e:
            return f"Error adding to queue: {e}"
