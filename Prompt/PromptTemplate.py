from langchain_core.prompts import PromptTemplate

#PromptTemplate
template = PromptTemplate(
    template= """
Please Summarize the research paper title "{PaperInput}" with follwing specifications:
Explaination Style: {StyleInput}
Explaination Length: {LengthInput}
1. Mathematical Details: 
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concept using simple, inutiative code snippets where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with:"Insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligend with proivided style and length.
""",
input_variables=["PaperInput", "StyleInput", "LengthInput"],
validate_template=True
)

template.save("template.json")