# Chat with your PDF using Gemini 🤖

This is a Streamlit application that allows you to chat with your PDF documents using Google's Gemini Pro model. You can upload a PDF, and the application will create a vector store of the document's content. You can then ask questions about the document, and the application will use the Gemini Pro model to generate answers based on the document's content.

## Features ✨

*   **Chat with your PDF:** Ask questions about your PDF documents in a conversational way. The application uses a Retrieval Augmented Generation (RAG) architecture to provide answers based on the content of your PDF.
*   **Google Gemini Pro:** Uses the latest Gemini Pro model from Google for high-quality, conversational answers.
*   **Chat History:** Remembers your previous questions and answers, allowing for a natural, back-and-forth conversation with your document.
*   **Local Embeddings Fallback:** Automatically falls back to a local Hugging Face embedding model (`all-MiniLM-L6-v2`) if the Google API rate limit is reached. This ensures that the application remains functional even when the Google API is unavailable.
*   **Easy to use:** Simple and intuitive user interface built with Streamlit, allowing you to upload a PDF and start chatting in seconds.

## Architecture 🏗️

This application follows a Retrieval Augmented Generation (RAG) architecture. Here's how it works:

1.  **PDF Processing:** When you upload a PDF, the application uses the `PyPDF` library to extract the text from the document.
2.  **Text Chunking:** The extracted text is then split into smaller chunks using `RecursiveCharacterTextSplitter` from LangChain. This is done to ensure that the text is small enough to be processed by the embedding model.
3.  **Embedding and Vector Store:** The text chunks are then converted into numerical representations (embeddings) using either the Google `embedding-001` model or the Hugging Face `all-MiniLM-L6-v2` model. These embeddings are then stored in a `FAISS` vector store. FAISS (Facebook AI Similarity Search) is a library that allows for efficient similarity search on a large collection of vectors.
4.  **Question Answering:** When you ask a question, the application first performs a similarity search on the vector store to find the most relevant text chunks from the PDF. These text chunks are then passed to the Gemini Pro model along with your question and the chat history. The model then uses this information to generate a relevant and accurate answer.

## Technology Choices 🛠️

*   **Streamlit:** We chose Streamlit for its simplicity and ease of use. It allows us to create a beautiful and interactive web application with just a few lines of Python code.
*   **LangChain:** LangChain is a powerful framework for building applications with large language models (LLMs). We use it for text splitting, creating prompts, and interacting with the Gemini Pro model.
*   **Google Gemini Pro:** Gemini Pro is a powerful and versatile LLM from Google. It's great for conversational AI and provides high-quality answers.
*   **FAISS:** FAISS is a library for efficient similarity search. It's perfect for finding the most relevant text chunks from the PDF when you ask a question.
*   **Hugging Face Embeddings:** We use Hugging Face embeddings as a fallback to ensure that the application remains functional even if the Google API is unavailable. The `all-MiniLM-L6-v2` model is a small and efficient model that provides good performance.
*   **PyPDF:** `PyPDF` is a simple and effective library for extracting text from PDF documents.

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
        GOOGLE_API_KEY="YOUR_API_KEY"
        ```

## Usage 🚀

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

2.  **Upload your PDF:**
    *   Open the application in your browser.
    *   Use the file uploader in the sidebar to upload your PDF document.
    *   Click the "Process" button to create the vector store.

3.  **Chat with your PDF:**
    *   Once the document is processed, you can start asking questions in the chat input.