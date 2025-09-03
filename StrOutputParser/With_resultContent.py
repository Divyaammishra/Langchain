from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

template1 = PromptTemplate(
    template= 'Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template= 'Generate a 5 line summary for the following text./n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)

print(result2.content)
