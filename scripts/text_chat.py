import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # NIA's code, one folder up

import utils.logger  # noqa: F401 - logs this session's turns to logs/nia.log
from agent.chat import AssistantModel, ensure_llm_server

ensure_llm_server()
assistant = AssistantModel()
text = "Me: Hey Nia!"
print(text)

while True:
    if text.lower() == "q":
        break
    print("Assistant:", assistant.respond(text))
    text = input("Me:")
