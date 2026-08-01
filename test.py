import torch
from agent.chat import AssistantModel
from agent.prompts.feedback import feedback_prompt

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
    result_txt = response.get("text_reply", "")
    if response.get("action_type") == "feedback":
        response = assistant.chat(feedback_prompt.format(response_text=result_txt))
        print(response)
        result_txt = response.get("text_reply", "")


    print("Assistant:",result_txt)
    text = input("Me:")