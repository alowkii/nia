import torch
from agent.chat import AssistantModel

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

assistant = AssistantModel(model="gemma3:4b")
text = "Me: Hey Nia!"
print(text)

while True:
    if text.lower() == "q":
        break
    response = assistant.chat(text)

    print(response)
    print(type(response))
    action_type = response["action_type"]
    result_txt = response["text_reply"]
    print("Assistant:",result_txt)
    text = input("Me:")