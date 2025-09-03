from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

schema = [
    ResponseSchema(name='fact 1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact 2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact 3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 fact about {topic} /n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)

f_result = parser.parse(result.content)

print(f_result)