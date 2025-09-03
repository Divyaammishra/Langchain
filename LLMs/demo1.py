from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

message = [
    HumanMessage(
        content=input("Ask Your Question... ")
    )
]

llm = ChatGroq(model='llama3-8b-8192')

response = llm.invoke(message)

output_Text = response.content
print(output_Text)