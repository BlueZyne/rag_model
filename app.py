import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Use langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# --- 1. Load API Key and Configure ---
# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the key is loaded
if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your .env file.")
    st.stop() # Stop the app if the key is missing
else:
    # Configure the Google AI client
    genai.configure(api_key=api_key)

# --- 2. Set up the Streamlit Page ---
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.header("Chat with your PDF using Gemini 🤖")

# --- 3. Add the File Uploader to the sidebar ---
with st.sidebar:
    st.subheader("Your Document")
    pdf_file = st.file_uploader("Upload your PDF document and click 'Process'", type="pdf")
    process_button = st.button("Process")

# --- 4. Main content area (for chat later) ---
st.subheader("Chat with your document")