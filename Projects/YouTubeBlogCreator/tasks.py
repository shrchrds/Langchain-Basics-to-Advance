from crewai import Task
from tools import yt_tool
from agents import blog_researcher, blog_writer

# Research Task
research_task = Task(
    description = (
        "Identify the video {topic}."
        "Get detailed information about the video from the channel"
    ),
    expected_ouput = "A comprehensive 3 paragraphs long report based on the {topic} of video content."
,
tools=[yt_tool],
agent=blog_researcher,
)

# Writing task with language model configuration

write_task = Task(
    description = ("Get the information from the YouTube channel on the topic {topic}."),
    expected_output = "Summarize the information from the YouTube Channel video on the topic {topic} and create the content for the Blog",
    tools = [yt_tool],
    agent = blog_writer,
    async_execution = False,
    output_file = 'new-blg-post.md'
)