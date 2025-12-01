import os
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
    
    def play_track(self, track_name):
        """Search and play a track"""
        try:
            results = self.sp.search(q=track_name, limit=1, type='track')
            
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                track = results['tracks']['items'][0]
                artists = ", ".join([artist['name'] for artist in track['artists']])
                self.sp.start_playback(uris=[track_uri])
                message = f"Successfully playing: {track['name']} by {artists}"
                return message
            else:
                message = f"Track '{track_name}' not found"
                return message
        except Exception as e:
            message = f"Error playing track: {e}"
            return message
    
    def play_playlist(self, playlist_name):
        """Search and play a playlist"""
        try:
            results = self.sp.search(q=playlist_name, limit=1, type='playlist')
            
            if results['playlists']['items']:
                playlist_uri = results['playlists']['items'][0]['uri']
                playlist = results['playlists']['items'][0]
                self.sp.start_playback(context_uri=playlist_uri)
                message = f"Successfully playing playlist: {playlist['name']}"
                return message
            else:
                message = f"Playlist '{playlist_name}' not found"
                return message
        except Exception as e:
            message = f"Error playing playlist: {e}"
            return message
    
    def play_album(self, album_name):
        """Search and play an album"""
        try:
            results = self.sp.search(q=album_name, limit=1, type='album')
            
            if results['albums']['items']:
                album_uri = results['albums']['items'][0]['uri']
                album = results['albums']['items'][0]
                artists = ", ".join([artist['name'] for artist in album['artists']])
                self.sp.start_playback(context_uri=album_uri)
                message = f"Successfully playing album: {album['name']} by {artists}"
                return message
            else:
                message = f"Album '{album_name}' not found"
                return message
        except Exception as e:
            message = f"Error playing album: {e}"
            return message
    
    def pause(self):
        """Pause playback"""
        try:
            self.sp.pause_playback()
            message = "Playback paused successfully"
            return message
        except Exception as e:
            message = f"Error pausing playback: {e}"
            return message
    
    def resume(self):
        """Resume playback"""
        try:
            self.sp.start_playback()
            message = "Playback resumed successfully"
            return message
        except Exception as e:
            message = f"Error resuming playback: {e}"
            return message
    
    def next(self):
        """Skip to next track"""
        try:
            self.sp.next_track()
            # Small delay to let Spotify update
            import time
            time.sleep(0.5)
            current = self.get_current_track()
            if current:
                message = f"Skipped to next track: {current}"
            else:
                message = "Skipped to next track"
            return message
        except Exception as e:
            message = f"Error skipping track: {e}"
            return message
    
    def previous(self):
        """Go to previous track"""
        try:
            self.sp.previous_track()
            # Small delay to let Spotify update
            import time
            time.sleep(0.5)
            current = self.get_current_track()
            if current:
                message = f"Went back to previous track: {current}"
            else:
                message = "Went back to previous track"
            return message
        except Exception as e:
            message = f"Error going to previous track: {e}"
            return message
    
    def set_volume(self, volume_percent):
        """Set volume (0-100)"""
        try:
            volume_percent = int(volume_percent)
            if 0 <= volume_percent <= 100:
                self.sp.volume(volume_percent)
                message = f"Volume set to {volume_percent}%"
                return message
            else:
                message = "Volume must be between 0 and 100"
                return message
        except ValueError:
            message = f"Invalid volume value: {volume_percent}. Must be a number between 0-100"
            return message
        except Exception as e:
            message = f"Error setting volume: {e}"
            return message
    
    def increase_volume(self, step=10):
        """Increase volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current['device']:
                current_volume = current['device']['volume_percent']
                new_volume = min(current_volume + step, 100)
                self.sp.volume(new_volume)
                message = f"Volume increased from {current_volume}% to {new_volume}%"
                return message
            else:
                message = "No active device found to increase volume"
                return message
        except Exception as e:
            message = f"Error increasing volume: {e}"
            return message
    
    def decrease_volume(self, step=10):
        """Decrease volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current['device']:
                current_volume = current['device']['volume_percent']
                new_volume = max(current_volume - step, 0)
                self.sp.volume(new_volume)
                message = f"Volume decreased from {current_volume}% to {new_volume}%"
                return message
            else:
                message = "No active device found to decrease volume"
                return message
        except Exception as e:
            message = f"Error decreasing volume: {e}"
            return message
    
    def shuffle(self, state=True):
        """Toggle shuffle mode"""
        try:
            self.sp.shuffle(state)
            status = "enabled" if state else "disabled"
            message = f"Shuffle {status} successfully"
            return message
        except Exception as e:
            message = f"Error toggling shuffle: {e}"
            return message
    
    def repeat(self, state='context'):
        """
        Set repeat mode
        state: 'track', 'context', or 'off'
        - 'track': repeat current track
        - 'context': repeat current context (playlist/album)
        - 'off': turn off repeat
        """
        try:
            if state in ['track', 'context', 'off']:
                self.sp.repeat(state)
                if state == 'track':
                    message = "Repeat mode set to: repeat current track"
                elif state == 'context':
                    message = "Repeat mode set to: repeat playlist/album"
                else:
                    message = "Repeat mode turned off"
                return message
            else:
                message = "Invalid repeat state. Use 'track', 'context', or 'off'"
                return message
        except Exception as e:
            message = f"Error setting repeat: {e}"
            return message
    
    def add_to_queue(self, track_name):
        """Add a track to the queue"""
        try:
            results = self.sp.search(q=track_name, limit=1, type='track')
            
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                track = results['tracks']['items'][0]
                artists = ", ".join([artist['name'] for artist in track['artists']])
                self.sp.add_to_queue(track_uri)
                message = f"Successfully added to queue: {track['name']} by {artists}"
                return message
            else:
                message = f"Track '{track_name}' not found"
                return message
        except Exception as e:
            message = f"Error adding to queue: {e}"
            return message