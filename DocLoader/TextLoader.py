from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

loader = TextLoader('sample.txt')

docs = loader.load()

prompt = PromptTemplate(
    template='Write the summary of the text file - \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))



