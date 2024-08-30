from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama 
import streamlit as st 

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful Assistant. Please response to users queries"),
    ("user", "Question:{question}")
]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# App Title

st.title("Q&A Chatbot With Ollama")


# Drop Down to select various Azure OpenAI models

llm = st.sidebar.selectbox("Select Open Source Model", ["llama3.1", "phi3", "gemma2"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main Interface for User Input

st.write("Ask any Question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the question ")