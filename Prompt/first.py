from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st 
import os
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama3-8b-8192"
)
st.header("Reaserch Tool")

PaperInput = st.selectbox( "Select Paper You Want to Summarize", [ "Select Paper...", "Attention Is All You Need", "ImageNet Classification with Deep Convolutional Neural Networks", "Playing Atari with Deep Reinforcement Learning", "Mastering the game of Go with deep neural network and tree search", "Generative Adversarial Network"])

StyleInput = st.selectbox( "Select Explaination Style you want", ["Select Style...", "Begineer Friendly", "Technical", "Code Oriented", "Mathematical"])

LengthInput = st.selectbox( "Select Explaination Length", ["Select Length...", "Short: 1-2 Paragraph", "Medium: 3-5 Paragraph", "Long: Detailed Explaination"])

template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "PaperInput" : PaperInput,
        "StyleInput" : StyleInput,
        "LengthInput" : LengthInput
        
    })
    st.write(result)