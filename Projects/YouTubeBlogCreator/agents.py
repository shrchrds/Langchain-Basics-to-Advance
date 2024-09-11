from crewai import Agent
from tools import yt_tool

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import  ChatGroq

os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It")

# Create Senior BLog Content Researcher

blog_researcher = Agent(
    role='Blog Researcher from YouTube Videos',
    goal = 'Get the relevant content for the topic {topic} from YouTube Channel',
    verbose=True,
    memory=True,
    backstory = (
        "Expert in understanding videos in AI, Data Science, Machine Learning"
    ),
    tools = [yt_tool],
    llm = llm,
    allow_delegation = True
)

# Create Senior Blog Write Agent with YouTube Tool

blog_writer = Agent(
    role='Blog Wrier',
    goal = 'Narrate compelling tech stories about the Video {topic} from YouTube Channel',
    verbose=True,
    memory=True,
    backstory = (
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner"
    ),
    tools = [yt_tool],
    llm = llm,
    allow_delegation = False
)

