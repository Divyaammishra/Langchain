from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
The historicity of the Mahabharata, an ancient Indian epic, is a subject of extensive and often passionate scholarly debate. This report provides a comprehensive examination of the available evidence, drawn from archaeological findings, astronomical calculations, geological data, and textual analysis.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 15,
    chunk_overlap= 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)
