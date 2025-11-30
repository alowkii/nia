import re
import json
import ollama
from .prompts.initial import initial_prompt
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
                if action_dict["action_keyword"] is not None:
                    spotify.play_track(action_dict["action_keyword"])
            


    def chat(self, user_msg):
        user_msg = user_msg

        # Add user message to chat history
        self.messages.append({"role": "user", "content": user_msg})

        # Stream model response
        stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True
        )

        full_reply = ""

        for chunk in stream:
            token = chunk["message"]["content"]
            full_reply += token

        print()  # newline at the end

        # Save assistant response into history
        self.messages.append({"role": "assistant", "content": full_reply})

        cleaned_dict = self.clean_json(full_reply)

        if cleaned_dict["action_type"] is not None:
            self.take_action(cleaned_dict)

        return cleaned_dict