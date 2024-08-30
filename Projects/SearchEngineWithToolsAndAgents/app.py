import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Arxiv and Wikipedia Tools
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search=DuckDuckGoSearchRun(name='Search')

# Streamlit App

st.title("Langsmith Documentation Chat with Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant",
        "content":"Hi I am assistant who can search on web. How can assist you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is Generative AI?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)    

    with st.chat_message("assistant"):
       st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
       response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
       st.session_state.messages.append({"role":"assistant","content":response})
       st.write(response)

    