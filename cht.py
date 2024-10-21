import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read and extract text from PDFs
def extract_pdf_text(pdf_files):
    full_text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            full_text += page.extract_text()
    return full_text

# Split the text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save FAISS index with embeddings
def create_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load conversational chain with custom prompts
def build_conversational_chain():
    prompt_template = """
    Use the provided context to answer the user's question as precisely as possible.
    If the answer is not in the context, state that explicitly.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    return chain

# Search for relevant text and respond to user query
def answer_user_query(user_query):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Enable dangerous deserialization as we're using a trusted source
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    # Perform similarity search on the vector store
    relevant_docs = vector_store.similarity_search(user_query)

    # Create QA chain and get response
    conversational_chain = build_conversational_chain()
    response = conversational_chain(
        {"input_documents": relevant_docs, "question": user_query},
        return_only_outputs=True
    )
    
    st.write("**Answer:**", response["output_text"])

# Main application logic
def main():
    st.set_page_config(page_title="RAG-based Chatbot", layout="wide")
    st.header("Chat with PDF using RAG 💬")
    
    st.sidebar.title("Upload PDF files")
    pdf_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    if st.sidebar.button("Process PDFs"):
        if pdf_files:
            with st.spinner("Processing PDFs..."):
                # Extract and process PDF text
                extracted_text = extract_pdf_text(pdf_files)
                text_chunks = split_text_into_chunks(extracted_text)
                create_vectorstore(text_chunks)
                st.sidebar.success("PDF processed and ready for questions!")
        else:
            st.sidebar.warning("Please upload at least one PDF.")

    # User query input field
    user_query = st.text_input("Ask a question based on the PDFs")

    if user_query:
        answer_user_query(user_query)

if __name__ == "__main__":
    main()
