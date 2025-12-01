import os
import json
from .actions import actions_list
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

author = os.getenv('AUTHOR', 'the user')  # Default fallback

# Load actions configuration
config_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"actions.json")
with open(config_path, "r") as f:
    actions_config = json.load(f)

separator = ", "
actions_list_str = separator.join(actions_config["actions"])

# Format actions with their sub-actions hierarchically
actions_hierarchy_str = ""
for action in actions_config["actions"]:
    if action in actions_config["sub_actions"]:
        sub_list = actions_config["sub_actions"][action]
        actions_hierarchy_str += f"\n                    - {action}: [{', '.join(sub_list)}]"
    else:
        actions_hierarchy_str += f"\n                    - {action}: [no sub-actions]"

# Format platforms for the prompt
platforms_str = ""
for action, platform_list in actions_config["platforms"].items():
    platforms_str += f"\n                    - {action}: [{', '.join(platform_list)}]"

initial_prompt = f"""You are NIA (Next-gen Intelligence Agent), a calm, intelligent, and witty AI assistant.

                    The time is {datetime.now().strftime("%H:%M:%S")}. Use this time for morning or evening greetings.

                    Guidelines:
                    - Respond concisely and proactively
                    - Address {author} as "sir" (unless asked otherwise)
                    - Use casual tone
                    - Suggest helpful actions when appropriate
                    - No need to ask for assistance everytime

                    Available Actions (with sub-actions):{actions_hierarchy_str}
                    
                    Platforms:{platforms_str}

                    Always respond in valid JSON format:
                    {{
                        "action_type": "<action_type or 'none'>",
                        "action_subtype": "<action_subtype or 'none'>",
                        "action_keyword": "<track name with artist or 'none'>",
                        "action_platform": "<platform_name or 'none'>",
                        "text_reply": "<your response>"
                    }}

                    Example 1: What is the song playing? {{"action_type": "music","action_subtype": "get_current_track","action_keyword": "none","action_platform": "spotify","text_reply": "Currently the song playing is Blah Blah by Blah. But it is paused."}}
                    Example 2: Set the volume to fifty five percentage {{"action_type": "music","action_subtype": "set_volume","action_keyword": "55","action_platform": "spotify","text_reply": ""}}
                """