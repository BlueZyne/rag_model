import streamlit as st
import os
from typing import List, Dict, Tuple
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
import logging
from datetime import datetime
import json
import uuid

# Import configuration
try:
    from config import (
        ModelConfig, DocumentConfig, EmbeddingConfig, 
        UIConfig, FeatureFlags, LogConfig, APIConfig
    )
except ImportError:
    # Fallback if config.py doesn't exist
    class ModelConfig:
        FLASH_MODEL = "models/gemini-2.0-flash-exp"
        PRO_MODEL = "models/gemini-2.0-flash-thinking-exp-1219"
        DEFAULT_TEMPERATURE = 0.7
        MIN_TEMPERATURE = 0.0
        MAX_TEMPERATURE = 1.0
        TOP_P = 0.85
        TOP_K = 40
        MAX_OUTPUT_TOKENS = 2048
    
    class DocumentConfig:
        MAX_FILE_SIZE_MB = 50
        MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
        CHUNK_SIZE = 1000
        CHUNK_OVERLAP = 200
        SIMILARITY_SEARCH_K = 4
    
    class EmbeddingConfig:
        MODEL_NAME = "all-MiniLM-L6-v2"
        CACHE_FOLDER = "./models"
    
    class UIConfig:
        PAGE_TITLE = "Chat with PDF - Scholar AI"
        PAGE_ICON = "🤖"
        LAYOUT = "wide"
        INITIAL_SIDEBAR_STATE = "expanded"
        MAX_CHAT_HISTORY_DISPLAY = 10
    
    class FeatureFlags:
        ENABLE_FEEDBACK = False  # Removed - doesn't improve model
        ENABLE_TOKEN_TRACKING = True
        ENABLE_TEMPERATURE_CONTROL = True
        ENABLE_CONVERSATION_SUMMARY = False  # Removed - not essential for clients
        ENABLE_DOCUMENT_FILTER = True
        ENABLE_STREAMING = True
    
    class APIConfig:
        COST_PER_1K_INPUT_TOKENS = 0.00015
        COST_PER_1K_OUTPUT_TOKENS = 0.0006
        CHARS_PER_TOKEN = 4

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants (from config) ---
MAX_FILE_SIZE_MB = DocumentConfig.MAX_FILE_SIZE_MB
MAX_FILE_SIZE_BYTES = DocumentConfig.MAX_FILE_SIZE_BYTES
CHUNK_SIZE = DocumentConfig.CHUNK_SIZE
CHUNK_OVERLAP = DocumentConfig.CHUNK_OVERLAP
DEFAULT_TEMPERATURE = ModelConfig.DEFAULT_TEMPERATURE
FLASH_MODEL = ModelConfig.FLASH_MODEL
PRO_MODEL = ModelConfig.PRO_MODEL

# --- 1. Load API Key and Configure ---
def load_api_key() -> str:
    """Load API key with fallback options"""
    try:
        load_dotenv(encoding='utf-8')
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Try direct environment variable if .env fails
            api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            logger.info("API key loaded successfully")
        else:
            logger.error("API key not found in environment")
        return api_key
    except Exception as e:
        logger.error(f"Error loading .env file: {str(e)}")
        st.error(f"Error loading .env file: {str(e)}")
        return None

api_key = load_api_key()

# Check if the key is loaded
if not api_key:
    st.error("🚨 GOOGLE_API_KEY not found! Please set it in your .env file.")
    st.info("Create a `.env` file in the project root with: `GOOGLE_API_KEY=your_api_key_here`")
    st.stop()  # Stop the app if the key is missing
else:
    # Configure the Google AI client
    genai.configure(api_key=api_key)
    logger.info("Google AI configured successfully")

# --- 2. Set up the Streamlit Page ---
st.set_page_config(
    page_title=UIConfig.PAGE_TITLE,
    page_icon=UIConfig.PAGE_ICON,
    layout=UIConfig.LAYOUT,
    initial_sidebar_state=UIConfig.INITIAL_SIDEBAR_STATE
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .source-citation {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .feedback-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Chat with your PDF using Scholar 🤖</h1>', unsafe_allow_html=True)

# Initialize session states
if 'documents' not in st.session_state:
    st.session_state.documents = {}  # Store documents with their names
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {}
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0

# New session states for enhanced features
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}  # Store feedback for each response
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = {'input': 0, 'output': 0}
if 'temperature' not in st.session_state:
    st.session_state.temperature = ModelConfig.DEFAULT_TEMPERATURE
if 'active_documents' not in st.session_state:
    st.session_state.active_documents = set()  # Documents to include in search
if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = None

# --- 3. Helper Functions ---
def validate_file_size(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file size"""
    try:
        file_size = uploaded_file.size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            return False, f"File size ({size_mb:.2f}MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB}MB)"
        return True, ""
    except Exception as e:
        logger.error(f"Error validating file size: {str(e)}")
        return False, f"Error validating file: {str(e)}"

def export_chat_history() -> str:
    """Export chat history as markdown"""
    if not st.session_state.chat_history:
        return "# Chat History\n\nNo messages yet."
    
    markdown = f"# Chat History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown += f"**Session ID:** {st.session_state.session_id}\n\n"
    
    # Add token usage if available
    if FeatureFlags.ENABLE_TOKEN_TRACKING and st.session_state.total_tokens['input'] > 0:
        total_input = st.session_state.total_tokens['input']
        total_output = st.session_state.total_tokens['output']
        estimated_cost = (total_input / 1000 * APIConfig.COST_PER_1K_INPUT_TOKENS + 
                         total_output / 1000 * APIConfig.COST_PER_1K_OUTPUT_TOKENS)
        markdown += f"**Total Tokens:** {total_input + total_output:,} (Input: {total_input:,}, Output: {total_output:,})\n"
        markdown += f"**Estimated Cost:** ${estimated_cost:.4f}\n\n"
    
    markdown += "---\n\n"
    
    for idx, (author, message) in enumerate(st.session_state.chat_history):
        icon = "👤" if author == "user" else "🤖"
        markdown += f"### {icon} {author.capitalize()}\n\n"
        markdown += f"{message}\n\n"
        
        # Add feedback if available
        if FeatureFlags.ENABLE_FEEDBACK and idx in st.session_state.feedback:
            feedback = st.session_state.feedback[idx]
            feedback_icon = "👍" if feedback == "positive" else "👎"
            markdown += f"*Feedback: {feedback_icon}*\n\n"
        
        markdown += "---\n\n"
    
    return markdown

def estimate_tokens(text: str) -> int:
    """Estimate token count from text"""
    return len(text) // APIConfig.CHARS_PER_TOKEN

def generate_conversation_summary(api_key: str) -> str:
    """Generate a summary of the conversation using Gemini"""
    if not st.session_state.chat_history:
        return "No conversation to summarize."
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=FLASH_MODEL,
            temperature=0.3,
            google_api_key=api_key
        )
        
        # Format conversation
        conversation_text = "\n".join([
            f"{author}: {message}" 
            for author, message in st.session_state.chat_history
        ])
        
        prompt_template = """
        Please provide a concise summary of the following conversation between a user and an AI assistant.
        Include:
        1. Main topics discussed
        2. Key questions asked
        3. Important insights or answers provided
        
        Conversation:
        {conversation}
        
        Summary:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["conversation"])
        chain = prompt | llm | StrOutputParser()
        
        summary = chain.invoke({"conversation": conversation_text})
        logger.info("Conversation summary generated")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

# --- 4. Add the File Uploader to the sidebar ---
def process_uploaded_files(uploaded_files):
    """Processes uploaded PDF files with enhanced error handling and statistics"""
    if not uploaded_files:
        return
    
    start_time = time.time()
    
    with st.spinner("📚 Processing documents..."):
        text_chunks_dict = {}
        stats = {}
        
        for pdf in uploaded_files:
            # Validate file size
            is_valid, error_msg = validate_file_size(pdf)
            if not is_valid:
                st.error(f"❌ {pdf.name}: {error_msg}")
                logger.warning(f"File size validation failed for {pdf.name}: {error_msg}")
                continue
            
            logger.info(f"Processing file: {pdf.name}")
            chunks, file_stats = get_pdf_text_and_chunks(pdf)
            
            if chunks:
                text_chunks_dict[pdf.name] = chunks
                stats[pdf.name] = file_stats
                logger.info(f"Successfully processed {pdf.name}: {len(chunks)} chunks created")
            else:
                logger.warning(f"No chunks created for {pdf.name}")
        
        if text_chunks_dict:
            create_vector_store(text_chunks_dict)
            st.session_state.document_stats = stats
            st.session_state.processing_time = time.time() - start_time
            logger.info(f"Processing completed in {st.session_state.processing_time:.2f} seconds")
        else:
            st.error("❌ No valid text could be extracted from any of the documents.")
            logger.error("No valid documents processed")

with st.sidebar:
    st.subheader("📁 Your Documents")
    pdf_files = st.file_uploader(
        "Upload PDF documents and start chatting", 
        type="pdf", 
        accept_multiple_files=True,
        key='pdf_files',
        on_change=lambda: process_uploaded_files(st.session_state.pdf_files),
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB per file"
    )
    
    # Show uploaded documents with statistics and filtering
    if 'pdf_files' in st.session_state and st.session_state.pdf_files:
        st.write("**Uploaded documents:**")
        
        # Initialize active documents if empty
        if not st.session_state.active_documents:
            st.session_state.active_documents = set([pdf.name for pdf in st.session_state.pdf_files])
        
        # Document filter (if feature enabled)
        if FeatureFlags.ENABLE_DOCUMENT_FILTER and len(st.session_state.pdf_files) > 1:
            st.write("**Filter documents for search:**")
            for pdf in st.session_state.pdf_files:
                is_active = st.checkbox(
                    f"📄 {pdf.name[:30]}...", 
                    value=pdf.name in st.session_state.active_documents,
                    key=f"doc_filter_{pdf.name}"
                )
                if is_active:
                    st.session_state.active_documents.add(pdf.name)
                else:
                    st.session_state.active_documents.discard(pdf.name)
        else:
            for pdf in st.session_state.pdf_files:
                st.write(f"📄 {pdf.name}")
        
        # Show statistics
        for pdf in st.session_state.pdf_files:
            if pdf.name in st.session_state.document_stats:
                stats = st.session_state.document_stats[pdf.name]
                with st.expander(f"📊 Statistics for {pdf.name[:20]}..."):
                    st.metric("Pages", stats.get('pages', 'N/A'))
                    st.metric("Text Chunks", stats.get('chunks', 'N/A'))
                    st.metric("Characters", f"{stats.get('characters', 0):,}")
        
        if st.session_state.processing_time > 0:
            st.info(f"⏱️ Processing time: {st.session_state.processing_time:.2f}s")
    
    st.divider()
    
    # Temperature Control (if feature enabled)
    if FeatureFlags.ENABLE_TEMPERATURE_CONTROL:
        st.subheader("🌡️ AI Settings")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=ModelConfig.MIN_TEMPERATURE,
            max_value=ModelConfig.MAX_TEMPERATURE,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher values make output more creative but less focused. Lower values make it more deterministic."
        )
        st.caption(f"Current: {st.session_state.temperature}")
        st.divider()
    
    # Token Usage Tracking (if feature enabled)
    if FeatureFlags.ENABLE_TOKEN_TRACKING and st.session_state.total_tokens['input'] > 0:
        st.subheader("📊 Usage Statistics")
        total_tokens = st.session_state.total_tokens['input'] + st.session_state.total_tokens['output']
        estimated_cost = (
            st.session_state.total_tokens['input'] / 1000 * APIConfig.COST_PER_1K_INPUT_TOKENS +
            st.session_state.total_tokens['output'] / 1000 * APIConfig.COST_PER_1K_OUTPUT_TOKENS
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col2:
            st.metric("Est. Cost", f"${estimated_cost:.4f}")
        
        with st.expander("Token Details"):
            st.write(f"**Input Tokens:** {st.session_state.total_tokens['input']:,}")
            st.write(f"**Output Tokens:** {st.session_state.total_tokens['output']:,}")
        
        st.divider()
    
    # Conversation Summary (if feature enabled)
    if FeatureFlags.ENABLE_CONVERSATION_SUMMARY and st.session_state.chat_history:
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating conversation summary..."):
                summary = generate_conversation_summary(api_key)
                st.session_state.conversation_summary = summary
        
        if st.session_state.conversation_summary:
            with st.expander("💡 Conversation Summary", expanded=False):
                st.write(st.session_state.conversation_summary)
        
        st.divider()
    
    # Export chat history button
    if st.session_state.chat_history:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Export Chat", use_container_width=True):
                chat_md = export_chat_history()
                st.download_button(
                    label="📥 Download",
                    data=chat_md,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        with col2:
            clear_button = st.button("🗑️ Clear All", use_container_width=True)
    else:
        clear_button = st.button("🗑️ Clear All")
    
    # Session info
    st.caption(f"Session ID: {st.session_state.session_id}")

# Handle clear functionality
if clear_button:
    st.session_state.documents = {}
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.session_state.pdf_files = []
    st.session_state.document_stats = {}
    st.session_state.processing_time = 0
    logger.info("Session cleared by user")
    st.rerun()

# --- 5. Function to Process the PDF ---
def get_pdf_text_and_chunks(pdf_file) -> Tuple[List[str], Dict]:
    """Extracts text from the PDF and splits it into chunks with statistics"""
    try:
        # Convert the uploaded file to a file-like object
        pdf_file_obj = io.BytesIO(pdf_file.read())
        pdf_file.seek(0)  # Reset file pointer
        
        text = ""
        pdf_reader = PdfReader(pdf_file_obj)
        page_count = len(pdf_reader.pages)
        
        logger.info(f"Reading {page_count} pages from {pdf_file.name}")
        
        # Extract text page by page with encoding error handling
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                # Clean and normalize text
                page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Could not extract text from page {page_num}: {str(e)}")
                st.warning(f"⚠️ Warning: Could not extract text from page {page_num} of {pdf_file.name}")
                continue
        
        if not text.strip():
            st.error(f"❌ No text could be extracted from {pdf_file.name}. Please check if the file is text-based and not scanned.")
            logger.error(f"No text extracted from {pdf_file.name}")
            return [], {}
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            logger.warning(f"No chunks created for {pdf_file.name}")
            st.warning(f"⚠️ Warning: No text chunks were created for {pdf_file.name}. The PDF might be empty or contain non-textual content.")
            return [], {}
        
        # Compile statistics
        stats = {
            'pages': page_count,
            'chunks': len(chunks),
            'characters': len(text)
        }
        
        logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
        return chunks, stats
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
        st.error(f"❌ Error processing {pdf_file.name}: {str(e)}")
        st.info("💡 Try saving your PDF with a different encoding or check if it's not corrupted.")
        return [], {}

# --- 6. Function to Create the Vector Store ---
def create_vector_store(text_chunks_dict: dict):
    """Creates a FAISS vector store from multiple documents' text chunks"""
    if not text_chunks_dict:
        st.error("❌ No text chunks to process")
        logger.error("No text chunks provided to create_vector_store")
        return

    try:
        logger.info("Creating embeddings and vector store")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder="./models"
        )
        
        # Process all documents and create a single vector store
        all_chunks = []
        all_metadatas = []
        
        for doc_name, chunks in text_chunks_dict.items():
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_name,
                    "chunk_id": chunk_idx
                })
        
        vector_store = FAISS.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=all_metadatas
        )
        st.session_state.vector_store = vector_store
        
        success_msg = f"✅ Successfully processed {len(text_chunks_dict)} document(s) with {len(all_chunks)} total chunks!"
        st.success(success_msg)
        logger.info(success_msg)
        
    except Exception as e:
        error_msg = f"Error creating vector store: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        if "memory" in str(e).lower():
            st.warning("💡 This might be a memory issue. Try uploading fewer or smaller documents.")
        return None

# --- 7. Handle User Questions ---
def get_question_complexity(question: str, api_key: str) -> str:
    """Classifies a question as simple or complex using the flash model"""
    try:
        llm = ChatGoogleGenerativeAI(
            model=FLASH_MODEL,
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
        
        result = chain.invoke({"question": question}).strip().lower()
        logger.info(f"Question classified as: {result}")
        return result
    except Exception as e:
        logger.warning(f"Could not classify question complexity: {str(e)}")
        return "simple"  # Default to simple if classification fails

# --- 8. Main Chat Interface ---
st.subheader("💬 Ask a question about your document")

# Display chat history
for author, message in st.session_state.chat_history:
    with st.chat_message(author):
        st.markdown(message)

if user_question := st.chat_input("Type your question here:"):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append(("user", user_question))
    logger.info(f"User question: {user_question}")

    if st.session_state.vector_store is not None:
        try:
            with st.spinner("🤔 Thinking..."):
                # Step 1: Classify the question
                complexity = get_question_complexity(user_question, api_key)

                # Step 2: Choose the model based on complexity
                if complexity == "complex":
                    model_name = PRO_MODEL
                    logger.info(f"Using PRO model for complex question")
                else:
                    model_name = FLASH_MODEL
                    logger.info(f"Using FLASH model for simple question")

                # Check for general greetings
                general_questions = ["hi", "hello", "thanks", "thank you", "hey"]
                if user_question.lower().strip() in general_questions:
                    context = ""
                    sources = ""
                    docs = []
                else:
                    vector_store = st.session_state.vector_store
                    docs = vector_store.similarity_search(user_question, k=DocumentConfig.SIMILARITY_SEARCH_K)
                    
                    # Filter docs by active documents if feature enabled
                    if FeatureFlags.ENABLE_DOCUMENT_FILTER and st.session_state.active_documents:
                        docs = [doc for doc in docs if doc.metadata["source"] in st.session_state.active_documents]
                    
                    context = "\n".join([doc.page_content for doc in docs])
                    sources = ", ".join(set([doc.metadata["source"] for doc in docs]))
                    logger.info(f"Retrieved {len(docs)} relevant chunks from sources: {sources}")
                
                # Track input tokens
                if FeatureFlags.ENABLE_TOKEN_TRACKING:
                    input_tokens = estimate_tokens(context + user_question)
                    st.session_state.total_tokens['input'] += input_tokens
                
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=st.session_state.temperature,  # Use dynamic temperature
                    top_p=ModelConfig.TOP_P,
                    top_k=ModelConfig.TOP_K,
                    max_output_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
                    google_api_key=api_key
                )
                
                prompt_template = """
                You are Scholar AI, a helpful and friendly AI assistant specialized in analyzing documents. 
                Your goal is to answer the user's question based on the provided context from their uploaded documents.
                
                If the user asks a question that is not related to the context, you can answer it in a friendly and conversational way.
                If the answer cannot be found in the context, politely explain that you are an AI assistant focused on answering questions about the provided documents.
                
                Always be accurate, concise, and helpful. When referencing information from the documents, be specific.

                Context: {context}
                Sources: {sources}
                Chat History: {chat_history}

                Question: {question}

                Answer:
                """
                prompt = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["context", "sources", "chat_history", "question"]
                )
                
                chain = prompt | llm | StrOutputParser()

                formatted_chat_history = "\n".join([
                    f'{author}: {message}' 
                    for author, message in st.session_state.chat_history[-UIConfig.MAX_CHAT_HISTORY_DISPLAY:]
                ])

                response = chain.invoke({
                    "context": context,
                    "sources": sources,
                    "chat_history": formatted_chat_history,
                    "question": user_question
                })
                
                # Track output tokens
                if FeatureFlags.ENABLE_TOKEN_TRACKING:
                    output_tokens = estimate_tokens(response)
                    st.session_state.total_tokens['output'] += output_tokens
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    # Show source citations if available
                    if sources:
                        st.markdown(f'<div class="source-citation">📚 <strong>Sources:</strong> {sources}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Add feedback buttons if feature enabled
                    if FeatureFlags.ENABLE_FEEDBACK:
                        response_idx = len(st.session_state.chat_history)
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"thumbs_up_{response_idx}"):
                                st.session_state.feedback[response_idx] = "positive"
                                st.success("Thanks for your feedback!")
                        with col2:
                            if st.button("👎", key=f"thumbs_down_{response_idx}"):
                                st.session_state.feedback[response_idx] = "negative"
                                st.info("Thanks for your feedback!")
                
                st.session_state.chat_history.append(("assistant", response))
                logger.info(f"Response generated successfully using {model_name}")

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            st.error("⚠️ Error generating response")
            st.error(f"Details: {str(e)}")
            
            # Provide helpful error messages
            if "quota" in str(e).lower() or "429" in str(e):
                st.warning("💡 This might be an API quota issue. Please check your API limits or try again later.")
            elif "timeout" in str(e).lower():
                st.warning("💡 Request timed out. Try asking a simpler question or check your internet connection.")
            elif "invalid" in str(e).lower() and "api" in str(e).lower():
                st.warning("💡 There might be an issue with your API key. Please verify it's correct in your .env file.")
    else:
        st.warning("⚠️ Please upload and process a PDF document first.")
        logger.warning("User attempted to ask question without uploading documents")

# --- 9. Footer ---
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🤖 Powered by Google Gemini")
with col2:
    st.caption("📚 RAG Architecture")
with col3:
    st.caption("⚡ Production Ready")