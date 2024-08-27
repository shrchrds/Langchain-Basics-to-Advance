import os 
from dotenv import load_dotenv

from langchain_community.llms import Ollama 
import streamlit as st 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are helpful assistant. Please answer to questions in friendly and professional manner"),
        ("user","Question:{question}")
    ]
)

# streamlit framework

st.title("Langchain Demo with Llama3.1")

input_text = st.text_input("What is in your mind?")


### Ollama Llama 3.1 Model

llm = Ollama(model="llama3.1")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    chain.invoke({"question": input_text})