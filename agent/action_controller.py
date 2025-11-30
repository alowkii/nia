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
                self.sp.start_playback(uris=[track_uri])
                print(f"Playing: {results['tracks']['items'][0]['name']}")
                return True
            else:
                print("Track not found")
                return False
        except Exception as e:
            print(f"Error playing track: {e}")
            return False
    
    def play_playlist(self, playlist_name):
        """Search and play a playlist"""
        try:
            results = self.sp.search(q=playlist_name, limit=1, type='playlist')
            
            if results['playlists']['items']:
                playlist_uri = results['playlists']['items'][0]['uri']
                self.sp.start_playback(context_uri=playlist_uri)
                print(f"Playing playlist: {results['playlists']['items'][0]['name']}")
                return True
            else:
                print("Playlist not found")
                return False
        except Exception as e:
            print(f"Error playing playlist: {e}")
            return False
    
    def play_album(self, album_name):
        """Search and play an album"""
        try:
            results = self.sp.search(q=album_name, limit=1, type='album')
            
            if results['albums']['items']:
                album_uri = results['albums']['items'][0]['uri']
                self.sp.start_playback(context_uri=album_uri)
                print(f"Playing album: {results['albums']['items'][0]['name']}")
                return True
            else:
                print("Album not found")
                return False
        except Exception as e:
            print(f"Error playing album: {e}")
            return False
    
    def pause(self):
        """Pause playback"""
        try:
            self.sp.pause_playback()
            print("Paused")
            return True
        except Exception as e:
            print(f"Error pausing: {e}")
            return False
    
    def resume(self):
        """Resume playback"""
        try:
            self.sp.start_playback()
            print("Resumed")
            return True
        except Exception as e:
            print(f"Error resuming: {e}")
            return False
    
    def next(self):
        """Skip to next track"""
        try:
            self.sp.next_track()
            print("Skipped to next track")
            return True
        except Exception as e:
            print(f"Error skipping track: {e}")
            return False
    
    def previous(self):
        """Go to previous track"""
        try:
            self.sp.previous_track()
            print("Previous track")
            return True
        except Exception as e:
            print(f"Error going to previous track: {e}")
            return False
    
    def set_volume(self, volume_percent):
        """Set volume (0-100)"""
        volume_percent = int(volume_percent)
        try:
            if 0 <= volume_percent <= 100:
                self.sp.volume(volume_percent)
                print(f"Volume set to {volume_percent}%")
                return True
            else:
                print("Volume must be between 0 and 100")
                return False
        except Exception as e:
            print(f"Error setting volume: {e}")
            return False
    
    def increase_volume(self, step=10):
        """Increase volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current['device']:
                current_volume = current['device']['volume_percent']
                new_volume = min(current_volume + step, 100)
                self.sp.volume(new_volume)
                print(f"Volume increased to {new_volume}%")
                return True
            else:
                print("No active device found")
                return False
        except Exception as e:
            print(f"Error increasing volume: {e}")
            return False
    
    def decrease_volume(self, step=10):
        """Decrease volume by specified step (default 10%)"""
        try:
            current = self.get_current_playback()
            if current and current['device']:
                current_volume = current['device']['volume_percent']
                new_volume = max(current_volume - step, 0)
                self.sp.volume(new_volume)
                print(f"Volume decreased to {new_volume}%")
                return True
            else:
                print("No active device found")
                return False
        except Exception as e:
            print(f"Error decreasing volume: {e}")
            return False
    
    def shuffle(self, state=True):
        """Toggle shuffle mode"""
        try:
            self.sp.shuffle(state)
            status = "enabled" if state else "disabled"
            print(f"Shuffle {status}")
            return True
        except Exception as e:
            print(f"Error toggling shuffle: {e}")
            return False
    
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
                print(f"Repeat set to: {state}")
                return True
            else:
                print("Invalid repeat state. Use 'track', 'context', or 'off'")
                return False
        except Exception as e:
            print(f"Error setting repeat: {e}")
            return False
    
    def get_current_playback(self):
        """Get current playback information"""
        try:
            return self.sp.current_playback()
        except Exception as e:
            print(f"Error getting playback info: {e}")
            return None
    
    def get_current_track(self):
        """Get currently playing track information"""
        try:
            current = self.sp.current_playback()
            if current and current['item']:
                track = current['item']
                artists = ", ".join([artist['name'] for artist in track['artists']])
                return {
                    'name': track['name'],
                    'artists': artists,
                    'album': track['album']['name'],
                    'is_playing': current['is_playing']
                }
            else:
                print("No track currently playing")
                return None
        except Exception as e:
            print(f"Error getting current track: {e}")
            return None
    
    def get_devices(self):
        """Get available devices"""
        try:
            devices = self.sp.devices()
            return devices['devices']
        except Exception as e:
            print(f"Error getting devices: {e}")
            return []
    
    def transfer_playback(self, device_id):
        """Transfer playback to a different device"""
        try:
            self.sp.transfer_playback(device_id)
            print(f"Transferred playback to device: {device_id}")
            return True
        except Exception as e:
            print(f"Error transferring playback: {e}")
            return False
    
    def add_to_queue(self, track_name):
        """Add a track to the queue"""
        try:
            results = self.sp.search(q=track_name, limit=1, type='track')
            
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                self.sp.add_to_queue(track_uri)
                print(f"Added to queue: {results['tracks']['items'][0]['name']}")
                return True
            else:
                print("Track not found")
                return False
        except Exception as e:
            print(f"Error adding to queue: {e}")
            return False