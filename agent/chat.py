import re
import json
import ollama
from .prompts.initial import initial_prompt
from .prompts.feedback import feedback_prompt
from .action_controller import SpotifyController

class AssistantModel:
    def __init__(self, model="gemma3:4b"):
        self.model = model
        self.messages = [
            {"role": "system", "content": initial_prompt}
        ]

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
        
        # Try parsing as-is first
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from text (look for { ... } or [ ... ])
        # Find the first { and last }
        json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON in code blocks (```json ... ```)
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_string, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fix common issues
        cleaned = json_string.strip()
        
        # Remove markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)
        
        # Remove trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Try again after cleaning
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unable to extract valid JSON from response. Error: {e}\nResponse: {json_string[:200]}")
        
    def take_action(self, action_dict):
        if action_dict["action_type"] == "music":
            if action_dict["action_platform"].lower() == "spotify":
                spotify = SpotifyController()
                if action_dict["action_subtype"] == 'play_track':
                    return spotify.play_track(action_dict["action_keyword"])
                elif action_dict["action_subtype"] == 'play_playlist':
                    return spotify.play_playlist(action_dict["action_keyword"])
                elif action_dict["action_subtype"] == 'play_album':
                    return spotify.play_album(action_dict["action_keyword"])
                elif action_dict["action_subtype"] == 'pause_track':
                    return spotify.pause()
                elif action_dict["action_subtype"] == 'resume_track':
                    return spotify.resume()
                elif action_dict["action_subtype"] == 'next_track':
                    return spotify.next()
                elif action_dict["action_subtype"] == 'previous_track':
                    return spotify.previous()
                elif action_dict["action_subtype"] == 'set_volume':
                    return spotify.set_volume(action_dict["action_keyword"])
                elif action_dict["action_subtype"] == 'decrease_volume':
                    return spotify.decrease_volume()
                elif action_dict["action_subtype"] == 'increase_volume':
                    return spotify.increase_volume()
                elif action_dict["action_subtype"] == 'shuffle':
                    return spotify.shuffle()
                elif action_dict["action_subtype"] == 'repeat_track':
                    return spotify.repeat("track")
                elif action_dict["action_subtype"] == 'repeat_context':
                    return spotify.repeat("context")
                elif action_dict["action_subtype"] == 'repeat_off':
                    return spotify.repeat("off")
                elif action_dict["action_subtype"] == 'add_to_queue':
                    return spotify.add_to_queue(action_dict["action_keyword"])
            

    def chat(self, user_msg):
        user_msg = user_msg

        # Add user message to chat history
        self.messages.append({"role": "user", "content": user_msg})

        # Stream model response
        stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
            format='json'
        )

        full_reply = ""

        for chunk in stream:
            token = chunk["message"]["content"]
            full_reply += token

        print()  # newline at the end

        # Save assistant response into history
        self.messages.append({"role": "assistant", "content": full_reply})

        cleaned_dict = self.clean_json(full_reply)

        if cleaned_dict["action_type"] is not None and str(cleaned_dict["action_type"]) != "none" and str(cleaned_dict["action_type"]) != "text_reply":
            feedback = self.take_action(cleaned_dict)
            if feedback:
                cleaned_dict["action_type"] = "feedback"
                cleaned_dict["text_reply"] = feedback
        
        return cleaned_dict