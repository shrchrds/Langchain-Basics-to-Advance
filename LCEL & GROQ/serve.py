from fastapi import FastAPI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import  ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

from langserve import add_routes

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="Gemma2-9b-It")

# create prompt template

system_template = "Translate the following in {language} language"

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)

parser = StrOutputParser()

# Create chain

chain = prompt_template | model | parser

# App definition

app = FastAPI(title="Langchain Server",
               version = "1.0",
               description = "Simple API Server using Langchain runnable interfaces" 
                )

# Adding Chain Routes

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)