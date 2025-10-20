import streamlit as st
import os
from typing import List
import google.generativeai as genai
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from dotenv import load_dotenv
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# --- 1. Load API Key and Configure ---
def load_api_key():
    """Load API key with fallback options"""
    try:
        load_dotenv(encoding='utf-8')
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Try direct environment variable if .env fails
            api_key = os.environ.get("GOOGLE_API_KEY")
        return api_key
    except Exception as e:
        st.error(f"Error loading .env file: {str(e)}")
        return None

api_key = load_api_key()

# Check if the key is loaded
if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your .env file.")
    st.stop()  # Stop the app if the key is missing
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

# --- 4. Function to Process the PDF ---
def get_pdf_text_and_chunks(pdf_file) -> List[str]:
    """Extracts text from the PDF and splits it into chunks."""
    try:
        # Convert the uploaded file to a file-like object
        pdf_file_obj = io.BytesIO(pdf_file.read())
        pdf_file.seek(0)  # Reset file pointer
        
        text = ""
        pdf_reader = PdfReader(pdf_file_obj)
        
        # Extract text page by page with encoding error handling
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                # Clean and normalize text
                page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                text += page_text + "\n"
            except Exception as e:
                st.warning(f"Warning: Could not extract text from a page: {str(e)}")
                continue
        
        if not text.strip():
            st.error("No text could be extracted from the PDF. Please check if the file is text-based and not scanned.")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            st.warning("Warning: No text chunks were created. The PDF might be empty or contain non-textual content.")
            return []
            
        return chunks
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        st.info("Try saving your PDF with a different encoding or check if it's not corrupted.")
        return []

# --- 5. Function to Create the Vector Store ---
def create_vector_store(text_chunks: List[str]):
    """Creates a FAISS vector store from text chunks and saves it to session state."""
    if not text_chunks:
        st.error("No text chunks to process")
        return

    try:
        # First try Google embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("Google API rate limit reached. Falling back to local embeddings...")
                # Fallback to HuggingFace embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    cache_folder="./models"
                )
            else:
                raise e

        # Add progress bar for chunks processing
        chunks_length = len(text_chunks)
        progress_bar = st.progress(0)
        
        # Process in smaller batches
        batch_size = 5
        all_embeddings = []
        
        for i in range(0, chunks_length, batch_size):
            batch = text_chunks[i:i + batch_size]
            vector_store = FAISS.from_texts(
                texts=batch,
                embedding=embeddings
            )
            
            if i == 0:
                final_store = vector_store
            else:
                final_store.merge_from(vector_store)
                
            # Update progress
            progress = min((i + batch_size) / chunks_length, 1.0)
            progress_bar.progress(progress)
            
            # Add small delay to avoid rate limits
            time.sleep(0.5)
        
        st.session_state.vector_store = final_store
        progress_bar.empty()
        st.success("Vector store created successfully!")
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        st.info("If you're seeing rate limit errors, try again in a few minutes")
        return None

# --- 6. The Main Logic ---
if process_button and pdf_file is not None:
    with st.spinner("Processing your document..."):
        # Get text chunks
        raw_text_chunks = get_pdf_text_and_chunks(pdf_file)
        
        # Create and save vector store
        create_vector_store(raw_text_chunks)
        
        st.success("✅ Document processed! You can now ask questions.")

# --- 7. Handle User Questions ---
st.subheader("Ask a question about your document")
user_question = st.text_input("Type your question here:")

# Initialize session state for vector store and chat history
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if user_question and "vector_store" in st.session_state:
    try:
        with st.spinner("Searching for answers..."):
            vector_store = st.session_state.vector_store
            if vector_store is None:
                st.error("Vector store is not properly initialized")
                st.stop()
                
            docs = vector_store.similarity_search(user_question, k=4)
            context = "\n".join([doc.page_content for doc in docs])
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
                top_p=0.85,
                top_k=40,
                max_output_tokens=2048,
                google_api_key=api_key
            )
            
            # Updated prompt template
            prompt_template = """
            You are a helpful AI assistant. Using the provided context, answer the user's question accurately and concisely.
            If the answer cannot be found in the context, simply state "I cannot find the answer in the provided context."
            
            Context: {context}
            
            Question: {question}
            
            Answer: Let me help you with that.
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            # Create the modern LCEL chain
            chain = prompt | llm | StrOutputParser()

            # Invoke the chain
            response = chain.invoke({"context": context, "question": user_question})
            
            # Display the answer
            st.write("### Answer")
            st.write(response)
    except Exception as e:
        st.error("⚠️ Error generating response")
        st.error(f"Details: {str(e)}")
        if "quota" in str(e).lower():
            st.warning("This might be an API quota issue. Please check your API limits.")