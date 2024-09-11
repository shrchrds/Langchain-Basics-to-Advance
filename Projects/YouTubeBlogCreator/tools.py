from crewai_tools import YoutubeChannelSearchTool

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import  ChatGroq

os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It")

yt_tool = YoutubeChannelSearchTool(youtube_channel_handle="@krishnaik06", llm=llm)

