from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

UserInput = input('Enter the Topic: ')

prompt1 = PromptTemplate(
    template='Write a joke about the topic - {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke in short - {text}',
    input_variables=['text']
)

JokeChain = RunnableSequence(prompt1, model, parser)

ParallelChain = RunnableParallel({
    'Joke':RunnablePassthrough(),
    'Explaination':RunnableSequence(prompt2, model, parser)
})

FinalChain = RunnableSequence(JokeChain, ParallelChain)
result = FinalChain.invoke({'topic':UserInput})
print(result['Joke'])
print(result['Explaination'])


