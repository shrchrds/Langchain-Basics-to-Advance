import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.summarize import load_summarize_chain
import logging
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From Website")
st.subheader('Summarize URL')

# Get the HuggingFace API Key and URL (website) to be summarized
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API token", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_kwargs = {"max_length": 512}
llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs=model_kwargs, temperature=0.5, huggingfacehub_api_token=hf_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from Website"):
    # Validate all the inputs
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load website data
                try:
                    response = requests.get(generic_url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    content = soup.get_text(separator="\n", strip=True)
                    logging.debug(f"Website content length: {len(content)}")
                except Exception as e:
                    logging.error(f"Failed to retrieve content from {generic_url}: {str(e)}")
                    content = ""

                if not content.strip():
                    st.error(f"No content was retrieved from the provided URL ({generic_url}). Please check the URL and try again.")
                else:
                    # Prepare the documents for summarization
                    documents = [Document(page_content=content)]

                    # Chain For Summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run({"input_documents": documents})

                    st.success(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error("Error occurred while processing the URL", exc_info=True)
