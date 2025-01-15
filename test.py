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
from serpapi import GoogleSearch

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="OpenCode: PovKet Assistant",
    page_icon="📈",
)

# Custom CSS for chat-like UI
st.markdown("""
    <style>
    .chat-container {
        overflow-y: auto;
        max-height: 600px;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: green;
        border-radius: 10px;
        padding: 10px;
        margin-left: auto;
        margin-bottom: 5px;
        align-self: flex-start;
        width: fit-content;
        max-width: 70%;
        word-wrap: break-word;
    }
    .bot-message {
        background-color: gray;
        border-radius: 10px;
        padding: 10px;
        align-self: flex-end;
        text-align: left;
        width: fit-content;
        max-width: 70%;
        word-wrap: break-word;
    }
    .sources {
        font-size: 0.8em;
        color: lightgray;
        margin-top: 5px;
        word-wrap: break-word;
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
        main_placeholder = st.empty()

        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Loading URLs...✅")
            docs.extend(loader.load())

        if uploaded_file:
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp_pdf.pdf")
            main_placeholder.text("Loading PDF...✅")
            docs.extend(loader.load_and_split())
            os.remove("temp_pdf.pdf")

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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat Container
chat_container = st.container()

# Input Container
input_container = st.container()

llm = OpenAI(temperature=0.9, max_tokens=500)
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_bytes):
    """Transcribe audio using OpenAI's whisper-1 model."""
    try:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
        )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def search_web(query, num_results=3):
    """Perform a web search using SerpAPI."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results

def get_openai_answer(query):
    """Get an answer from OpenAI's API."""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=500,
        temperature=0.9
    )
    return response.choices[0].text.strip()

def query_openai_model(query):
    """Query OpenAI's text-davinci-003 model directly."""
    try:
        response = openai_client.completions.create(
            model="text-davinci-003",
            prompt=query,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error querying OpenAI: {e}")
        return None

def query_gemini(query):
    """Query Gemini API for an answer."""
    # Ensure you have the 'google-generativeai' library installed: pip install google-generativeai
    import google.generativeai as genai
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # Ensure you have this in your .env
    if not GOOGLE_API_KEY:
        st.error("GEMINI_API_KEY not found in environment variables.")
        return None
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    try:
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        st.error(f"Error querying Gemini API: {e}")
        return None

# Function to display messages from chat history
def display_chat_history():
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            is_user = message["is_user"]
            message_type = "user-message" if is_user else "bot-message"

            with st.container():
                st.markdown(
                    f'<div class="{message_type}">{message["text"]}</div>',
                    unsafe_allow_html=True
                )
                if "sources" in message and message["sources"]:
                    st.markdown(f'<div class="sources">{message["sources"]}</div>', unsafe_allow_html=True)

# Function to add message to history
def add_message_to_history(text, is_user, sources=None):
    st.session_state.chat_history.append({"text": text, "is_user": is_user, "sources": sources})

# Initial display of chat history
display_chat_history()

# Text input and audio input logic within the input container
with input_container:
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input("wassup, how can i be of help➰:", key="text_input")
        if query:
            add_message_to_history(query, is_user=True)
            try:
                #Process only the last question
                if os.path.exists(faiss_index_path):
                    vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                    answer = result["answer"]
                    sources = result.get("sources", "")
                    if not answer or answer.lower() in ["i do not know", ""]:
                        answer = query_gemini(query)
                        if not answer or answer.lower() in ["i do not know", ""] or "election" in query.lower() and not "2024" in answer.lower():
                            search_results = search_web(query, num_results=5)
                            if search_results and "organic_results" in search_results:
                                for result in search_results["organic_results"]:
                                    title = result.get("title", "")
                                    snippet = result.get("snippet", "")
                                    link = result.get("link", "")
                                    bot_response = f"**{title}**\n{snippet}\n[Link]({link})"
                                    add_message_to_history(bot_response, is_user=False)
                                    break
                            else:
                                add_message_to_history("No relevant information found.", is_user=False)
                        else:
                            add_message_to_history(answer, is_user=False)
                    else:
                        add_message_to_history(answer, is_user=False, sources=sources)
                else:
                    answer = query_gemini(query)
                    if not answer or answer.lower() in ["i do not know", ""] or "election" in query.lower() and not "2024" in answer.lower():
                        search_results = search_web(query, num_results=5)
                        if search_results and "organic_results" in search_results:
                            for result in search_results["organic_results"]:
                                title = result.get("title", "")
                                snippet = result.get("snippet", "")
                                link = result.get("link", "")
                                bot_response = f"**{title}**\n{snippet}\n[Link]({link})"
                                add_message_to_history(bot_response, is_user=False)
                                break
                        else:
                            add_message_to_history("No relevant information found.", is_user=False)
                    else:
                        add_message_to_history(answer, is_user=False)
            except Exception as e:
                add_message_to_history(f"An error occurred: {e}", is_user=False)
            except Exception as e:
                add_message_to_history(f"An error occurred: {e}", is_user=False)
            except Exception as e:
                add_message_to_history(f"An error occurred: {e}", is_user=False)
            display_chat_history()

    with col2:
        audio_bytes = st.audio_input(label="Record", key="audio_input")
        if audio_bytes:
            st.text("Transcribing Audio...🎙️")
            try:
                query = transcribe_audio(audio_bytes)
                add_message_to_history(query, is_user=True)
                if os.path.exists(faiss_index_path):
                    vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                    answer = result["answer"]
                    sources = result.get("sources", "")

                    if not answer or answer.lower() in ["i do not know", ""]:
                        # If no answer is found in the FAISS index, use OpenAI's API
                        answer = get_openai_answer(query)
                        if not answer or answer.lower() in ["i do not know", ""]:
                            # If OpenAI's API also does not provide a satisfactory answer, perform a web search
                            search_results = search_web(query, num_results=5)
                            if search_results and "organic_results" in search_results:
                                for result in search_results["organic_results"]:
                                    title = result.get("title", "")
                                    snippet = result.get("snippet", "")
                                    link = result.get("link", "")
                                    bot_response = f"**{title}**\n{snippet}\n[Link]({link})"
                                    add_message_to_history(bot_response, is_user=False)
                            else:
                                add_message_to_history("No relevant search results found.", is_user=False)
                        else:
                            add_message_to_history(answer, is_user=False)
                    else:
                        add_message_to_history(answer, is_user=False, sources=sources)
                else:
                    add_message_to_history("FAISS index not found. Please process data first.", is_user=False)
            except Exception as e:
                add_message_to_history(f"Error transcribing audio: {e}", is_user=False)
            display_chat_history()
