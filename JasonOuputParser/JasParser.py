from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and, city of a fictional character /n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions}
)

prompt = template.format()

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)


#With Chains 
#chain = template | model | parser
#result = chain.invoke({})
#print(result)