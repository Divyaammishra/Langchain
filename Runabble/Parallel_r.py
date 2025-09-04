from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

prompt1 = PromptTemplate(
    template='Write a creative post for twitter on the topic - {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write a creative post for linkedin on the topic - {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

ParallelChain = RunnableParallel({
    'Tweet':RunnableSequence(prompt1, model, parser),
    'LinkedIn':RunnableSequence(prompt2, model, parser)
})

UserInput = input('Enter the Topic: ')
result = ParallelChain.invoke({'topic':UserInput})

print(result['Tweet'])
print(result['LinkedIn'])