# Chat with your PDF using Gemini 🤖

This is a Streamlit application that allows you to chat with your PDF documents using Google's Gemini Pro model. You can upload a PDF, and the application will create a vector store of the document's content. You can then ask questions about the document, and the application will use the Gemini Pro model to generate answers based on the document's content.

## Features ✨

*   **Chat with your PDF:** Ask questions about your PDF documents in a conversational way.
*   **Google Gemini Pro:** Uses the latest Gemini Pro model from Google for high-quality answers.
*   **Chat History:** Remembers your previous questions and answers, so you can have a conversation with your document.
*   **Local Embeddings Fallback:** Automatically falls back to a local embedding model if the Google API rate limit is reached.
*   **Easy to use:** Simple and intuitive user interface built with Streamlit.

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

## Technologies Used 🛠️

*   **Streamlit:** For the web application framework.
*   **LangChain:** For the language model integration and text processing.
*   **Google Gemini Pro:** As the language model for generating answers.
*   **FAISS:** For creating the vector store.
*   **Hugging Face Embeddings:** As a fallback for creating embeddings.
*   **PyPDF:** For extracting text from PDF documents.
