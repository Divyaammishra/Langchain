from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Strings of Fire.pdf')

docs = loader.load()

print(docs)