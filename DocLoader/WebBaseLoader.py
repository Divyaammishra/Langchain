from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

url='https://academy.openai.com/home'
loader = WebBaseLoader(url)

docs = loader.load()

UserInput = input('Ask me anything: ')

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text \n {text}',
    input_variables=['question','text']
)
 
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':UserInput,'text':docs[0].page_content})

print(result)