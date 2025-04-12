import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from functools import lru_cache

# Constants with smaller models
HUGGINGFACE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_TASK = "question-answering"
HUGGINGFACE_MODEL_QA = "deepset/roberta-base-squad2"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Cached models
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)

@lru_cache(maxsize=1)
def get_qa_pipeline():
    return pipeline(task=HUGGINGFACE_TASK, model=HUGGINGFACE_MODEL_QA)

# Streamlit UI
st.header("My first Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf", accept_multiple_files=False)

if file is not None:
    # Check file size
    if file.size > MAX_FILE_SIZE:
        st.error(f"File too large. Please upload a PDF smaller than {MAX_FILE_SIZE//(1024*1024)}MB.")
        st.stop()
    
    with st.spinner('Processing PDF...'):
        # Extract text
        pdf_reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings
        embeddings = get_embeddings()
        vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Type your question here")

    if user_question:
        with st.spinner('Finding answers...'):
            try:
                match = vector_store.similarity_search(user_question, k=3)  # Limit to top 3 matches
                context = " ".join([doc.page_content for doc in match])
                
                question_answerer = get_qa_pipeline()
                response = question_answerer(question=user_question, context=context)
                
                st.write(f"**Answer:** {response['answer']}")
                st.write(f"**Confidence:** {response['score']:.2f}")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try a different question or check your document.")