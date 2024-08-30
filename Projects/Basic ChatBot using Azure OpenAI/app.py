import streamlit as st 
import openai
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_ENDPOINT")

llm = AzureChatOpenAI(model="gpt-4o",api_version="2024-05-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT)

def generate_response(question, api_key=AZURE_OPENAI_API_KEY, llm=llm, temperature=0.7, max_tokens=150, azure_endpoint=AZURE_OPENAI_ENDPOINT):
    openai.api_key = api_key
    llm = AzureChatOpenAI(model=llm,api_version="2024-05-01-preview", azure_endpoint=AZURE_OPENAI_ENDPOINT)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer


# App Title

st.title("Q&A Chatbot With Azure OpenAI")

# Sidebar for Settings

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Azure OpenAI API Key", type="password")

# Drop Down to select various Azure OpenAI models

llm = st.sidebar.selectbox("Select Azure OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5"])

# Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main Interface for User Input

st.write("Ask any Question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the question ")
