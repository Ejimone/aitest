import os
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
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.core.audio import SoundLoader
from kivy.properties import StringProperty, ListProperty, ObjectProperty
from dotenv import load_dotenv

load_dotenv()

class ChatbotApp(App):
    chat_history = ListProperty([])
    input_text = StringProperty("")

    def build(self):
        root = BoxLayout(orientation='vertical')
        self.chat_log = ScrollView(do_scroll_x=False)
        self.chat_log_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.chat_log_layout.bind(minimum_height=self.chat_log_layout.setter('height'))
        self.chat_log.add_widget(self.chat_log_layout)
        root.add_widget(self.chat_log)

        input_layout = BoxLayout(orientation='horizontal')
        self.user_input = TextInput(multiline=False, size_hint_x=0.8)
        send_button = Button(text='Send', on_press=self.send_message)
        input_layout.add_widget(self.user_input)
        input_layout.add_widget(send_button)
        root.add_widget(input_layout)
        return root

    def load_documents(self, urls, pdf_path):
        docs = []
        if urls:
            loader = UnstructuredURLLoader(urls=urls)
            try:
                docs.extend(loader.load())
            except Exception as e:
                return f"Error loading URLs: {e}"
        if pdf_path:
            loader = PyPDFLoader(pdf_path)
            try:
                docs.extend(loader.load())
            except Exception as e:
                return f"Error loading PDF: {e}"
        return docs

    def send_message(self, instance):
        user_message = self.user_input.text
        self.add_message_to_history(user_message, is_user=True)
        self.user_input.text = ""
        llm = OpenAI(temperature=0.9, max_tokens=500)
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        faiss_index_path = "faiss_index_openai"

        def query_gemini(query):
            import google.generativeai as genai
            GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GOOGLE_API_KEY:
                return "GEMINI_API_KEY not found in environment variables."
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            try:
                response = model.generate_content(query)
                return response.text
            except Exception as e:
                return f"Error querying Gemini API: {e}"

        try:
            if os.path.exists(faiss_index_path):
                vectorstore = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": user_message}, return_only_outputs=True)
                bot_response = result["answer"]
            else:
                bot_response = "FAISS index not found. Please process data first."
        except Exception as e:
            bot_response = f"An error occurred: {e}"

        self.add_message_to_history(bot_response, is_user=False)
        self.update_chat_log()

    def add_message_to_history(self, text, is_user):
        self.chat_history.append({"text": text, "is_user": is_user})

    def update_chat_log(self):
        self.chat_log_layout.clear_widgets()
        for message in self.chat_history:
            message_label = Label(text=message["text"], markup=True)
            message_label.text_size = (Window.width * 0.7, None)
            message_label.halign = "left" if message["is_user"] else "right"
            self.chat_log_layout.add_widget(message_label)

if __name__ == '__main__':
    ChatbotApp().run()
