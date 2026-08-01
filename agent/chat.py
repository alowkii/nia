import re
import json
import ollama
from datetime import datetime
from .prompts.initial import initial_prompt
from .action_controller import SpotifyController

class AssistantModel:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.messages = [
            {"role": "system", "content": initial_prompt}
        ]
        self._spotify = None  # built on first music action, not at startup

    def clean_json(self, json_string):
        """
        Extract and parse JSON from a response that may contain additional text.

        Args:
            json_string (str): The response text that may contain JSON

        Returns:
            dict: Parsed JSON object
        """
        if not json_string or not json_string.strip():
            raise ValueError("Response is empty or contains only whitespace")

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass

        # Grab the outermost { ... }, which also strips ```json fences
        match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Unable to extract valid JSON from response: {json_string[:200]}")

    def spotify(self):
        if self._spotify is None:
            self._spotify = SpotifyController()
        return self._spotify

    def take_action(self, action_dict):
        """Run the requested action, returning a feedback string (or None if unhandled)."""
        if action_dict.get("action_type") != "music":
            return None
        if str(action_dict.get("action_platform", "")).lower() != "spotify":
            return None

        sp = self.spotify()
        keyword = action_dict.get("action_keyword")
        handlers = {
            "play_track": lambda: sp.play_track(keyword),
            "play_playlist": lambda: sp.play_playlist(keyword),
            "play_album": lambda: sp.play_album(keyword),
            "pause_track": sp.pause,
            "resume_track": sp.resume,
            "next_track": sp.next,
            "previous_track": sp.previous,
            "set_volume": lambda: sp.set_volume(keyword),
            "decrease_volume": sp.decrease_volume,
            "increase_volume": sp.increase_volume,
            "shuffle": sp.shuffle,
            "repeat_track": lambda: sp.repeat("track"),
            "repeat_context": lambda: sp.repeat("context"),
            "repeat_off": lambda: sp.repeat("off"),
            "add_to_queue": lambda: sp.add_to_queue(keyword),
            "get_current_track": sp.now_playing,
        }
        handler = handlers.get(action_dict.get("action_subtype"))
        return handler() if handler else None

    def chat(self, user_msg):
        # Refresh the clock every turn - this process stays up for days
        self.messages[0]["content"] = (
            f"The current time is {datetime.now().strftime('%H:%M:%S')}.\n\n{initial_prompt}"
        )

        # Add user message to chat history
        self.messages.append({"role": "user", "content": user_msg})

        reply = ollama.chat(
            model=self.model,
            messages=self.messages,
            format='json'
        )
        full_reply = reply["message"]["content"]

        # Save assistant response into history
        self.messages.append({"role": "assistant", "content": full_reply})

        cleaned_dict = self.clean_json(full_reply)

        action_type = str(cleaned_dict.get("action_type") or "none").lower()
        if action_type not in ("none", "text_reply"):
            feedback = self.take_action(cleaned_dict)
            if feedback:
                cleaned_dict["action_type"] = "feedback"
                cleaned_dict["text_reply"] = feedback

        return cleaned_dict
