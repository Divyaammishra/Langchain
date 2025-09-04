from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

UserInput= input('Enter the Topic: ')

JokePrompt = PromptTemplate(
    template='Write a joke on the topic - {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

JokeChain = RunnableSequence(JokePrompt, model, parser)

Chain = RunnableParallel({
    'Joke':RunnablePassthrough(),
    'WordCounter':RunnableLambda(lambda x: len(x.split()))
})

FinalChain = RunnableSequence(JokeChain, Chain)

result = FinalChain.invoke({'topic':UserInput})

print(result)
