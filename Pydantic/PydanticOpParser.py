from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Optional
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

class Person(BaseModel):

    name: str = Field(description='Name of the Person')
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person) 

template = PromptTemplate(
    template='Generate the name, age and, city of the fictonal {place} character /n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'place':'India'})

result = model.invoke(prompt)

final_r = parser.parse(result.content)

print(final_r)