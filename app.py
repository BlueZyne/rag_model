import streamlit as st
import os
from typing import List
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
st.header("Chat with your PDF using Scholar 🤖")

# Initialize session states
if 'documents' not in st.session_state:
    st.session_state.documents = {}  # Store documents with their names
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Add the File Uploader to the sidebar ---
with st.sidebar:
    st.subheader("Your Documents")
    pdf_files = st.file_uploader("Upload PDF documents and click 'Process'", type="pdf", accept_multiple_files=True)
    
    # Show uploaded documents
    if pdf_files:
        st.write("Uploaded documents:")
        for pdf in pdf_files:
            st.write(f"📄 {pdf.name}")
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("Process All")
    with col2:
        clear_button = st.button("Clear All")

# Handle clear functionality
if clear_button:
    st.session_state.documents = {}
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.rerun()

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
def create_vector_store(text_chunks_dict: dict):
    """Creates a FAISS vector store from multiple documents' text chunks."""
    if not text_chunks_dict:
        st.error("No text chunks to process")
        return

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./models"
        )
        
        # Process all documents and create a single vector store
        all_chunks = []
        all_metadatas = []
        
        for doc_name, chunks in text_chunks_dict.items():
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": doc_name})
        
        vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=all_metadatas
        )
        st.session_state.vector_store = vector_store
        st.success(f"✅ Successfully processed {len(text_chunks_dict)} documents!")
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# --- 6. The Main Logic ---
if process_button and pdf_files:
    with st.spinner("Processing documents..."):
        text_chunks_dict = {}
        for pdf in pdf_files:
            chunks = get_pdf_text_and_chunks(pdf)
            if chunks:  # Only add if chunks were successfully created
                text_chunks_dict[pdf.name] = chunks
        
        if text_chunks_dict:
            create_vector_store(text_chunks_dict)
        else:
            st.error("No valid text could be extracted from any of the documents.")

# --- 7. Handle User Questions ---
def get_question_complexity(question: str, api_key: str) -> str:
    """Classifies a question as simple or complex using the flash model."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0.0,
            google_api_key=api_key
        )
        
        prompt_template = """
        Classify the following question as "simple" or "complex". 
        A "simple" question is one that can be answered with a short, factual answer.
        A "complex" question is one that requires more explanation, reasoning, or a multi-part answer.
        Your response should be a single word: "simple" or "complex".

        Question: {question}

        Classification:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
        
        chain = prompt | llm | StrOutputParser()
        
        return chain.invoke({"question": question}).strip().lower()
    except Exception as e:
        st.warning(f"Could not classify question complexity: {str(e)}")
        return "simple"  # Default to simple if classification fails

st.subheader("Ask a question about your document")
 
 # Initialize session state for vector store and chat history
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
 
 # Display chat history
for author, message in st.session_state.chat_history:
    with st.chat_message(author):
        st.markdown(message)
 
if user_question := st.chat_input("Type your question here:"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append(("user", user_question))

    if "vector_store" in st.session_state and st.session_state.vector_store is not None:
        try:
            with st.spinner("Thinking..."):
                # Step 1: Classify the question
                complexity = get_question_complexity(user_question, api_key)

                # Step 2: Choose the model based on complexity
                if complexity == "complex":
                    model_name = "gemini-2.5-pro"
                    pass
                else:
                    model_name = "models/gemini-2.5-flash"
                    pass

                general_questions = ["hi", "hello", "thanks", "thank you"]
                if user_question.lower() in general_questions:
                    context = ""
                    sources = ""
                else:
                    vector_store = st.session_state.vector_store
                    docs = vector_store.similarity_search(user_question, k=4)
                    context = "\n".join([doc.page_content for doc in docs])
                    sources = ", ".join(set([doc.metadata["source"] for doc in docs]))
                
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0.7,
                    top_p=0.85,
                    top_k=40,
                    max_output_tokens=2048,
                    google_api_key=api_key
                )
                
                prompt_template = """
                You are a helpful and friendly AI assistant. Your goal is to answer the user's question based on the provided context.
                If the user asks a question that is not related to the context, you can answer it in a friendly and conversational way.
                If the answer cannot be found in the context, you can say that you are an AI assistant and your primary purpose is to answer questions about the provided documents.

                Context: {context}
                Sources: {sources}
                Chat History: {chat_history}

                Question: {question}

                Answer:
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "sources", "chat_history", "question"])
                
                chain = prompt | llm | StrOutputParser()

                formatted_chat_history = "\n".join([f'{author}: {message}' for author, message in st.session_state.chat_history])

                response = chain.invoke({
                    "context": context,
                    "sources": sources,
                    "chat_history": formatted_chat_history,
                    "question": user_question
                })
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.chat_history.append(("assistant", response))

        except Exception as e:
            st.error("⚠️ Error generating response")
            st.error(f"Details: {str(e)}")
            if "quota" in str(e).lower():
                st.warning("This might be an API quota issue. Please check your API limits.")
    else:
        st.warning("Please upload and process a PDF document first.")