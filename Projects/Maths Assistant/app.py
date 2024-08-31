import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain 
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


import os 
from dotenv import load_dotenv
load_dotenv()

# Load GROQ API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="gemma2-9b-it")
# Set up streamlit app
st.set_page_config(page_title="Text to Math Problem Solver")
st.title("Text to Math Problem Solver Using Gemma2")

# Initialize toools
wikipedia_wrapper =WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func = wikipedia_wrapper.run,
    description="A tool for searching the internet to find various topics mentioned"
)

# Initialize the Math Tool

math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions only"
)

prompt = """
    You are an agent tasked for solving users mathematical questions. Logically arrive at the solutions and display it point wise for the question below.
    Question: {question}
    Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into chain 
chain = LLMChain(llm=llm, prompt = prompt_template)

reasoning_tool = Tool(
    name="Reasining tool",
    func = chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# initialize the agents

assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True  
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
        "content":"Hi, I am Math Assistant. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


# Interaction
question = st.text_area("Enter your Maths related question")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response"):
            st.session_state.messages.append({'role':'user', 'content':question})
            st.chat_message('user').write(question)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role':'assistant', 'content':response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please Enter Maths Question")
