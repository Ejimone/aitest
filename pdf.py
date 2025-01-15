import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("OpenCode: Povket Assistant 📈")
st.sidebar.title("Input Data")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

process_data_clicked = st.sidebar.button("Process Data")

faiss_index_path = "faiss_index_openai"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_data_clicked:
    docs = []

    if urls:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading URLs...✅")
        docs.extend(loader.load())

    if uploaded_file:
        with open("temp_pdf.pdf", "wb") as f:  # Save the uploaded file temporarily
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp_pdf.pdf") # Load from the temporary file
        main_placeholder.text("Loading PDF...✅")
        docs.extend(loader.load_and_split())
        os.remove("temp_pdf.pdf")  # Remove the temporary file

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
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



query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
