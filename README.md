# My Chatbot 💬

This is a simple chatbot application built using Streamlit 🚀, LangChain 🔗, and Hugging Face Transformers 🤗.

**Functionality:**

* 📤 Allows users to upload PDF files.
* 📄 Extracts the text content from the uploaded PDF.
* ✂️ Splits the text into smaller chunks.
* 🧠 Generates vector embeddings for these chunks using a Hugging Face sentence transformer model (`sentence-transformers/all-mpnet-base-v2`).
* 🗂️ Creates a FAISS vector store to index the embeddings.
* ❓ Enables users to ask questions related to the content of the uploaded PDF.
* 🔍 Performs a similarity search on the vector store to find relevant document chunks.
* 🤖 Uses a Hugging Face question answering model (`deepset/roberta-large-squad2`) to extract answers from the relevant context.
* ✅ Displays the answer and a confidence score (if provided by the QA model).

**How to Use:**

1.  **Upload a PDF file:** Use the file uploader in the sidebar to select and upload a PDF document.
2.  **Ask a question:** Once the PDF is processed, a text input box will appear in the main area. Type your question related to the content of the PDF and press Enter.
3.  **View the answer:** The chatbot will process your question and display the extracted answer below the input box, along with a confidence score (if available).

**Dependencies:**

The application relies on the following Python libraries:

* `streamlit`: For creating the web interface.
* `PyPDF2`: For reading PDF files.
* `langchain-community`: For text splitting, embeddings, vector store (FAISS), and potentially other integrations.
* `transformers`: The core library from Hugging Face for accessing pre-trained models and pipelines.
* `sentence-transformers`: Specifically used for generating document embeddings.
* `faiss-cpu`: For efficient similarity search.

**Deployment:**

This application can be deployed using Streamlit Community Cloud by hosting the code (including `app.py` and `requirements.txt`) on a public GitHub repository.

**Repository Contents:**

* `app.py`: The main Python script containing the Streamlit application code.
* `requirements.txt`: A file listing the Python dependencies required to run the application.

** Live Demo:**
🔗 [https://genai-projects-sapna.streamlit.app/](https://genai-projects-sapna.streamlit.app/)

**Further Development:**

Potential future enhancements could include:

* ➕ Support for other document types (e.g., TXT, CSV).
* 💡 More sophisticated question answering strategies.
* ✨ Improved user interface and feedback.
* 📚 Handling of multiple uploaded documents.
* 💾 Saving and loading the vector store for faster processing.

**Author**
👩‍💻 Sapna Devi
