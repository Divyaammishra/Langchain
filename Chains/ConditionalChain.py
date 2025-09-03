from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

StringParser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['Positive', 'Negative'] = Field(description='Give the sentiment of the feedback')

PydanticParser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the following feedback into positive and negative \n {feedback} \n {format_instruction}',
    input_variables={'feedback'},
    partial_variables={'format_instruction':PydanticParser.get_format_instructions()}
)

ClassifierChain = prompt1 | model | PydanticParser

prompt2 = PromptTemplate(
    template='Write an appropriate response for this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response for this negative feedback \n {feedback}',
    input_variables=['feedback']
)

BranchChain = RunnableBranch(
    (lambda x:x.sentiment == 'Positive', prompt2 | model | StringParser),
    (lambda x:x.sentiment == 'Negative', prompt3 | model | StringParser),
    RunnableLambda(lambda x: "Could not classify feedback")
)

chain = ClassifierChain | BranchChain

print(chain.invoke({'feedback':'This is good phone'}))