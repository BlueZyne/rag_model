# Chat with your PDF using Gemini 🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ireadpdf.streamlit.app/)

**[View the live application](https://ireadpdf.streamlit.app/)**

This is a production-ready Streamlit application that allows you to chat with your PDF documents using Google's Gemini models. Upload PDFs, and the application creates a vector store of the document's content. Ask questions, and the application uses Gemini AI to generate accurate answers based on your documents.

## Features ✨

### Core Features
*   **Chat with your PDF:** Ask questions about your PDF documents in a conversational way using Retrieval Augmented Generation (RAG) architecture.
*   **Google Gemini AI:** Uses Gemini 2.0 Flash and Thinking models for high-quality, intelligent answers with automatic complexity-based routing.
*   **Multi-Document Support:** Upload and chat with multiple PDF documents simultaneously.
*   **Chat History:** Maintains conversation context for natural, back-and-forth dialogue.

### Production Features
*   **📊 Document Statistics:** View page count, text chunks, and character count for each uploaded document.
*   **💾 Export Chat History:** Download your conversations as markdown files with session metadata, token usage, and feedback.
*   **📚 Source Citations:** See which documents were used to answer your questions.
*   **⚡ File Validation:** Automatic file size validation (max 50MB) to prevent crashes.
*   **🔍 Smart Model Routing:** Automatically uses Flash model for simple questions and Pro model for complex queries.
*   **📝 Comprehensive Logging:** Full logging system for debugging and monitoring.
*   **🎨 Enhanced UI:** Beautiful gradient design with statistics cards and better user experience.
*   **⚠️ Better Error Handling:** Detailed, helpful error messages with troubleshooting suggestions.

### New Advanced Features
*   **🌡️ Temperature Control:** Adjust AI creativity and randomness with an easy-to-use slider (0.0-1.0).
*   **📊 Token Usage Tracking:** Real-time tracking of API token usage with cost estimation.
*   **👍👎 Response Feedback:** Rate AI responses with thumbs up/down to track quality.
*   **📝 Conversation Summary:** Generate AI-powered summaries of your chat sessions.
*   **🔍 Document Filtering:** Select which documents to include in your searches (multi-document mode).
*   **⚙️ Configurable Settings:** Centralized configuration file for easy customization.
*   **🔒 Enhanced Security:** Comprehensive security policies and best practices.

### Technical Features
*   **Local Embeddings:** Uses Hugging Face `all-MiniLM-L6-v2` model for efficient vector embeddings.
*   **FAISS Vector Store:** Fast similarity search for retrieving relevant document chunks.
*   **Robust Text Processing:** Handles encoding errors and extracts text from complex PDFs.
*   **Feature Flags:** Enable/disable features easily via configuration.

## Architecture 🏗️

This application follows a Retrieval Augmented Generation (RAG) architecture:

1.  **PDF Processing:** Extracts text from uploaded PDFs using `PyPDF` library with error handling.
2.  **Text Chunking:** Splits text into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`.
3.  **Embedding and Vector Store:** Converts chunks to embeddings using `all-MiniLM-L6-v2` and stores in FAISS vector store with metadata.
4.  **Question Classification:** Analyzes question complexity to route to appropriate Gemini model.
5.  **Question Answering:** Performs similarity search, retrieves relevant chunks, and generates answers using Gemini with chat history context.

## Technology Stack 🛠️

*   **Streamlit:** Interactive web application framework
*   **LangChain:** Framework for building LLM applications (text splitting, prompts, chains)
*   **Google Gemini 2.0:** Advanced AI models (Flash for speed, Thinking for complex reasoning)
*   **FAISS:** Efficient similarity search library
*   **Hugging Face Embeddings:** `all-MiniLM-L6-v2` for text embeddings
*   **PyPDF:** PDF text extraction
*   **Python Logging:** Production-grade logging system

## Installation and Setup ⚙️

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BlueZyne/rag_model.git
    cd rag_model
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv third
    source third/bin/activate  # On Windows, use `third\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    *   Create a `.env` file in the root of the project.
    *   Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY=your_api_key_here
        ```
    *   Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Usage 🚀

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Upload your PDF:**
    *   Open the application in your browser (usually `http://localhost:8501`)
    *   Use the file uploader in the sidebar to upload PDF documents (max 50MB each)
    *   Documents are automatically processed when uploaded
    *   View statistics for each document in the sidebar

3.  **Configure AI Settings (Optional):**
    *   Adjust the **Temperature** slider (0.0-1.0) to control AI creativity
        *   Lower (0.0-0.3): More focused and deterministic responses
        *   Medium (0.4-0.7): Balanced creativity and accuracy
        *   Higher (0.8-1.0): More creative and varied responses

4.  **Filter Documents (Multi-Document Mode):**
    *   When multiple PDFs are uploaded, use checkboxes to select which documents to search
    *   Only selected documents will be used to answer your questions

5.  **Chat with your PDF:**
    *   Once documents are processed, ask questions in the chat input
    *   The AI will automatically choose the best model based on question complexity
    *   View source citations to see which documents were referenced
    *   Rate responses with 👍 or 👎 to provide feedback

6.  **Monitor Usage:**
    *   View real-time token usage and estimated costs in the sidebar
    *   Track your API consumption across the session

7.  **Generate Conversation Summary:**
    *   Click "📝 Generate Summary" to get an AI-powered summary of your chat
    *   View key topics, questions, and insights from your conversation

8.  **Export your conversation:**
    *   Click "💾 Export Chat" in the sidebar
    *   Download your chat history as a markdown file with metadata
    *   Includes session ID, token usage, costs, and feedback ratings

## Troubleshooting 🔧

### Common Issues

**"GOOGLE_API_KEY not found"**
- Ensure you've created a `.env` file in the project root
- Verify the API key is correctly formatted: `GOOGLE_API_KEY=your_key_here`
- Check that the `.env` file is in the same directory as `app.py`

**"No text could be extracted from the PDF"**
- The PDF might be scanned/image-based. Try using OCR software first
- The PDF might be corrupted. Try opening it in a PDF reader
- The PDF might be password-protected. Remove protection first

**"API quota exceeded"**
- You've hit your Google AI API rate limit
- Wait a few minutes and try again
- Check your quota at [Google AI Studio](https://makersuite.google.com/)

**"File size exceeds maximum"**
- Maximum file size is 50MB per PDF
- Try compressing the PDF or splitting it into smaller files

**Memory errors**
- Try uploading fewer documents at once
- Restart the application to clear memory
- Close other applications to free up RAM

## Production Deployment 🚀

This application is production-ready with:
- Comprehensive error handling and logging
- File validation and size limits
- Rate limiting protection
- Efficient memory management
- User-friendly error messages

For deployment to Streamlit Cloud, Heroku, or other platforms:
1. Ensure `.env` is in `.gitignore` (it is by default)
2. Set `GOOGLE_API_KEY` as an environment variable in your deployment platform
3. The app will automatically use environment variables if `.env` is not found

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is open source and available under the MIT License.

## Acknowledgments 🙏

- Google for the Gemini AI models
- Streamlit for the amazing web framework
- LangChain for the RAG framework
- The open-source community for all the amazing tools