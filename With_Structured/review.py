from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192"
)

class Review(TypedDict):

    summary: str
    sentiment: str

StructuredModel = model.with_structured_output(Review)

result = StructuredModel.invoke(""" The "Cosmo" smartwatch is a great all-around device. It's comfortable, stylish, and offers solid performance for the price. The fitness tracking and battery life are excellent, though I found the app selection to be a bit limited. I'd definitely recommend it for anyone looking for a quality, affordable smartwatch.""")

print(result)