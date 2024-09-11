from crewai import Agent, Task
import os
from dotenv import load_dotenv
from crewai import Crew, Process
from langchain_openai import AzureChatOpenAI
from langchain_groq import  ChatGroq

load_dotenv()

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_ENDPOINT")

default_llm = AzureChatOpenAI(model="gpt-4o", api_version="2024-05-01-preview")

os.environ['OPENAI_API_KEY'] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Gemma2-9b-It")

# Create a researcher agent
researcher = Agent(
    role='Senior Researcher',
    goal='Discover groundbreaking technologies',
    verbose=True,
    llm=llm,
    backstory='A curious mind fascinated by cutting-edge innovation and the potential to change the world, you know everything about tech.'
)

# Task for the researcher with expected_output
research_task = Task(
    description='Identify the next big trend in AI',
    expected_output='A detailed analysis of the next big trend in AI, including potential implications and examples.',
    agent=researcher  # Assigning the task to the researcher
)

# Instantiate your crew
tech_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential  # Tasks will be executed one after the other
)

# Begin the task execution
tech_crew.kickoff()