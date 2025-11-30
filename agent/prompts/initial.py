import os
import json
from .actions import actions_list
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

author = os.getenv('AUTHOR', 'the user')  # Default fallback

# Load actions configuration
actions_config = {
    "actions": ["music", "shutdown"],
    "sub_actions": {
        "music": ["pause_music", "increase_volume", "decrease_volume", "play_music", "next_track", "previous_track", "shuffle", "repeat"]
    },
    "platforms": {
        "music": ["spotify", "local"]
    }
}

separator = ", "
actions_list_str = separator.join(actions_config["actions"])

# Format sub-actions for the prompt
sub_actions_str = ""
for action, sub_list in actions_config["sub_actions"].items():
    sub_actions_str += f"\n                    - {action}: {', '.join(sub_list)}"

# Format platforms for the prompt
platforms_str = ""
for action, platform_list in actions_config["platforms"].items():
    platforms_str += f"\n                    - {action} platforms: {', '.join(platform_list)}"

initial_prompt = f"""You are NIA (Next-gen Intelligence Agent), a calm, intelligent, and witty AI assistant.

                    The time is {datetime.now().strftime("%H:%M:%S")}. Use this time for morning or evening greetings.

                    Guidelines:
                    - Respond concisely and proactively
                    - Address {author} as "sir" (unless asked otherwise)
                    - Use casual tone
                    - Suggest helpful actions when appropriate
                    - No need to ask for assistance everytime

                    Available action_type: {actions_list_str}
                    
                    Available action_subtype: {sub_actions_str}
                    
                    Platforms:{platforms_str}

                    Always respond in valid JSON format:
                    {{
                        "action_type": "<action_type or 'none'>",
                        "action_subtype": "<action_subtype or 'none'>",
                        "action_keyword": "<track name with artist or 'none'>",
                        "action_platform": "<platform_name or 'none'>",
                        "text_reply": "<your response>"
                    }}
                """