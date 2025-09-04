from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableBranch, RunnablePassthrough
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

UserInput= input('Enter the Topic: ')

prompt = PromptTemplate(
    template='Write a detailed report on the topic - {topic}',
    input_variables=['topic']
)

ConditionalPrompt = PromptTemplate(
    template='Summarize the following text - {text}',
    input_variables=['text']
)

parser = StrOutputParser()

ReportChain = RunnableSequence(prompt, model, parser)

ConditionalChain = RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(ConditionalPrompt, model, parser)),
    RunnablePassthrough()
)

FinalChain = RunnableSequence(ReportChain, ConditionalChain)

print(FinalChain.invoke({'topic':UserInput}))