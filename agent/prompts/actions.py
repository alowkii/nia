import json
from pathlib import Path

actions_path = Path(__file__).parent.parent / "actions.json"

with open(actions_path, 'r', encoding='utf-8') as f:
    actions_dict = json.load(f)

actions_list = actions_dict["actions"]