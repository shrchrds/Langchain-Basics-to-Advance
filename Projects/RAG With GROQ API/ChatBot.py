import streamlit as st 
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACE_API_TOKEN'] = os.getenv("HUGGINGFACE_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Set up streamlit app

st.title('Conversation RAG With PDF Upload')
st.write("Upload PDFs to Chat")

llm = ChatGroq(model_name="Gemma2-9b-It")

# Chat Interface
session_id = st.text_input("Session ID", value="default_session")

# Manage chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose PDF File", type="pdf", accept_multiple_files=True)

# Process Uploaded Files

if uploaded_files:
    documents = []
    for file in uploaded_files:
        tempPDF = f"./temp.pdf"
        with open(tempPDF, "wb") as f:
            f.write(file.getvalue())
            file_name = file.name

        loader = PyPDFLoader(tempPDF)
        docs = loader.load()
        documents.extend(docs)
    
    # Split and Create embeddings for documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        """
            Given a chat history and the latest user question
            which might reference context in the chat history, 
            formulate a standalone question which can be understood.
            Without chat history, do NOT answer the question,
            just reformulate it if needed otherwise return it as it is.    
        """   
    )

    contextualize_q_prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer Question 
    system_prompt = (
        """
        You are an assistant for question answering task.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't kow. 
        Use three sentences maximum and keep answer concise.
        \n\n
        {context}
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    Conversational_RAG_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Ask Question to Chat with PDF: ")
    if user_input:
        session_history = get_session_history(session_id)
        response = Conversational_RAG_chain.invoke(
            {"input": user_input},
            config = {
                "configurable": {"session_id":session_id}
            },
        )

        # st.write(st.session_state.store)
        st.write("Assistant: ", response['answer'])

