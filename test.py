from agent.chat import AssistantModel

assistant = AssistantModel()
text = "Me: Hey Nia!"
print(text)

while True:
    if text.lower() == "q":
        break
    print("Assistant:", assistant.respond(text))
    text = input("Me:")
