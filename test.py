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
