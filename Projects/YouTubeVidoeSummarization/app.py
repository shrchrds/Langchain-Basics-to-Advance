import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
import youtube_dl
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or a website URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load video data using youtube-dl
                if "youtube.com" in generic_url:
                    ydl_opts = {
                        'format': 'best',
                        'noplaylist': True,
                        'quiet': True,
                        'force_generic_extractor': True,
                    }
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(generic_url, download=False)
                        video_title = info_dict.get('title', 'No Title Available')
                        video_description = info_dict.get('description', 'No Description Available')

                    # Combine title and description for summarization
                    content = f"Title: {video_title}\n\nDescription: {video_description}"
                else:
                    # Load website data
                    content = ""
                    # You can implement a method to scrape the website content if needed
                    # For example, using requests and BeautifulSoup

                # Check if content is empty
                if not content.strip():
                    st.error("No content was retrieved from the provided URL. Please check the URL and try again.")
                else:
                    # Prepare the documents for summarization
                    documents = [{"text": content}]  # Wrap content in a list of dictionaries

                    # Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run({"input_documents": documents})  # Use the correct key

                    st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error("Error occurred", exc_info=True)