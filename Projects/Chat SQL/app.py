import streamlit as st 
from pathlib import Path 
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

# Load GROQ API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="Chat With SQL DB: ")
st.title("Chat with SQL in Natural Language")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["USE SQLLITE 3 DATABASE Sales.db", "Connect to your SQL Database"]

selected_opt = st.sidebar.radio(label="Choose DB for Chat", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL 
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

if not db_uri:
    st.info("Please enter databse and URI")

# LLM Model

llm = ChatGroq(model="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_DB(db_uri, mysql_host=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        DBFilePath = (Path(__file__).parent/"Sales.db").absolute()

        creator = lambda:sqlite3.connect(f"file:{DBFilePath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator)) 

    elif db_uri ==MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))


if db_uri==MYSQL:
    db=configure_DB(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)

else:
    db = configure_DB(db_uri)

# Toolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [{"role": "assistant", "content":"How can I help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from Database")

if user_query:
    st.session_state.messages.append({"role": "user", "content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(input=user_query, callbacks=[streamlit_callback], handle_parsing_errors=True)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except ValueError as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, I couldn't process your request. Please try again."})