from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192"
)

ChatHistory = [
    SystemMessage(content="You are a knowledgeable AI assistant")
]

while True:
    UserInput = input("You: ")
    ChatHistory.append(HumanMessage(content=UserInput))
    if UserInput == 'exit':
        break
    result = model.invoke(ChatHistory)
    ChatHistory.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print(ChatHistory)