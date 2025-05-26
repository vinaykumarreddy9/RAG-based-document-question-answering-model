import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


groq_api_key = os.getenv('GROQ_API_KEY')
load_dotenv()

working_directory = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    documents = PyPDFLoader(file_path).load()
    return documents


def setup_vectorstores(documents):
    embeddings = OllamaEmbeddings(model='llama2')
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks,embeddings)
    return vectorstore

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(groq_api_key = groq_api_key ,model = 'llama-3.3-70b-versatile',temperature=0)

    memory = ConversationBufferMemory(
        llm = llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        verbose = True
    )

    return chain


st.set_page_config(
    page_title='chat_with_doc',
    page_icon='📝',
    layout='centered'
)

st.title("🦙 Chat with Doc - LLAMA 3.1")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label='upload your PDF file', type=['pdf'])

if uploaded_file:
    file_path = f"{working_directory}/{uploaded_file.name}"
    
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = setup_vectorstores(load_document(file_path=file_path))

    if 'chain' not in st.session_state:
        st.session_state.chain = create_chain(st.session_state.vectorstore)


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input("Ask LLAMA..... ")

if user_input:
    st.session_state.chat_history.append({'role' : 'user', "content" : user_input})

    with st.chat_message('user'):
        st.markdown(user_input)


    with st.chat_message('assistant'):
        response = st.session_state.chain({'question': user_input})
        assist_res = response['answer']
        st.markdown(assist_res)
        st.session_state.chat_history.append({"role" : 'assistant', "content" : assist_res})