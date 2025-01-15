import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import json

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="OpenCode Pocket Assistant",
    page_icon="📈",
)

# Custom CSS for chat-like UI
st.markdown("""
    <style>
    .chat-container {
        overflow-y: auto;
        # max-height: 600px; /* Adjust as needed */
    }
    .user-message {
        background-color: lightblue;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        align-self: flex-end;
        max-width: 70%;
    }
    .bot-message {
        background-color: lemon; # 
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        align-self: flex-start;
        max-width: 70%;
    }
    .sources {
        font-size: 0.8em;
        color: gray;
        margin-top: 5px;
    }
    .stTextArea textarea {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# Sidebar for data input
with st.sidebar:
    st.title("Input Data")
    urls = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}")
        urls.append(url)

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_data_clicked = st.button("Process Data")
    faiss_index_path = "faiss_index_openai"

    if process_data_clicked:
        docs = []
        main_placeholder = st.empty()  # create a placeholder on the sidebar

        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Loading URLs...✅")
            docs.extend(loader.load())

        if uploaded_file:
            with open("temp_pdf.pdf", "wb") as f:  # Save the uploaded file temporarily
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp_pdf.pdf")  # Load from the temporary file
            main_placeholder.text("Loading PDF...✅")
            docs.extend(loader.load_and_split())
            os.remove("temp_pdf.pdf")  # Remove the temporary file

        if docs:
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','], chunk_size=1000
            )
            main_placeholder.text("Splitting Text...✅")
            docs = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Building Embeddings...✅")

            vectorstore_openai.save_local(faiss_index_path)
            main_placeholder.text("FAISS index saved...✅")
        else:
            main_placeholder.text("Please provide URLs or upload a PDF.")

# Main app area
st.title("OpenCode: Pocket Assistant 📈")

# Chat Container
chat_container = st.container()

# Input Container
input_container = st.container()

llm = OpenAI(temperature=0.9, max_tokens=500)

def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI's whisper-1 model."""
    client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
    )
    return transcript.text


# Function to display messages
def display_message(message, is_user, sources=None):
  with chat_container:
        alignment = "flex-end" if is_user else "flex-start"
        message_type = "user-message" if is_user else "bot-message"

        st.markdown(f"""
            <div style="display: flex; justify-content: {alignment};">
              <div class="{message_type}">
                {message}
              </div>
            </div>
        """, unsafe_allow_html=True)
        if sources:
            st.markdown(f"""
              <div style="display: flex; justify-content: {alignment};">
              <div class="sources">
                  {sources}
              </div>
            </div>
        """,unsafe_allow_html=True)

# Text input and audio input logic within the input container
with input_container:
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input("Type your question here:", key="text_input")
        if query:
            if os.path.exists(faiss_index_path):
                vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                answer = result["answer"]
                sources = result.get("sources", "")

                display_message(query, is_user=True)
                display_message(answer, is_user=False, sources=sources)


    with col2:
        audio_bytes = st.audio_input(label="Record", key = "audio_input")
        if audio_bytes:
                st.text("Transcribing Audio...🎙️")
                try:
                    query = transcribe_audio(audio_bytes)
                except Exception as e:
                    st.error(f"Error transcribing audio: {e}")
                    query = None

                if query:
                    if os.path.exists(faiss_index_path):
                        vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                        result = chain({"question": query}, return_only_outputs=True)
                        answer = result["answer"]
                        sources = result.get("sources", "")

                        display_message(query, is_user=True)
                        display_message(answer, is_user=False, sources=sources)