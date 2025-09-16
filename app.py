# import packages
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from datetime import datetime

import asyncio
import nest_asyncio

nest_asyncio.apply()  # patch for Streamlit + grpc

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Store embeddings in FAISS
def get_vector_store(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Conversational chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say "answer is not available in the context". 
    Do not make up information. 

    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# MCQ Generator
def mcq_chain_function(llm, page_text):
    mcq_template = PromptTemplate(
        input_variables=["text"],
        template="""
You are an expert teacher. Generate 3 multiple-choice questions (with 4 options each) based on the following text.
Mark the correct answer.

Text: {text}

Output format:
Q1: ...
a) ...
b) ...
c) ...
d) ...
Answer: ...
"""
    )
    mcq_chain = LLMChain(llm=llm, prompt=mcq_template)
    mcqs = mcq_chain.run(text=page_text)
    return mcqs.strip()

# SAQ Generator
def saq_chain_function(llm, page_text):
    saq_template = PromptTemplate(
        input_variables=["text"],
        template="""
You are an expert teacher. Generate 3 short-answer questions based on the following text.

Text: {text}

Output format:
Q1: ...
Q2: ...
Q3: ...
"""
    )
    saq_chain = LLMChain(llm=llm, prompt=saq_template)
    saqs = saq_chain.run(text=page_text)
    return saqs.strip()


# Handle user input
def user_input(user_question, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDFs and enter your OpenAI API key.")
        return

    # Build vector store if not exists
    if not os.path.exists("faiss_index"):
        text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
        vector_store = get_vector_store(text_chunks, api_key)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Search for relevant docs
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Save to history
    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
    conversation_history.append(
        (user_question_output, response_output, "OpenAI", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names))
    )

    # ✅ Show conversation history only once (latest included)
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <style>
                .chat-message {{
                    padding: 1.5rem;
                    border-radius: 0.5rem;
                    margin-bottom: 1rem;
                    display: flex;
                }}
                .chat-message.user {{
                    background-color: #2b313e;
                }}
                .chat-message.bot {{
                    background-color: #475063;
                }}
                .chat-message .avatar {{
                    width: 20%;
                }}
                .chat-message .avatar img {{
                    max-width: 78px;
                    max-height: 78px;
                    border-radius: 50%;
                    object-fit: cover;
                }}
                .chat-message .message {{
                    width: 80%;
                    padding: 0 1.5rem;
                    color: #fff;
                }}
            </style>
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Download conversation as CSV
    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download Conversation as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)


# ========== MAIN APP ==========

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (OpenAI) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin_profile_link = "https://www.linkedin.com/in/arijit-dey-0784512a9/"
    kaggle_profile_link = "https://www.kaggle.com/arijitdey9430"
    github_profile_link = "https://github.com/Arijit4598"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_profile_link}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle_profile_link}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github_profile_link})"
    )

    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    st.sidebar.markdown("Get your API key from [OpenAI Platform](https://platform.openai.com/).")

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
            api_key = None
            pdf_docs = None
        elif clear_button:
            if 'user_question' in st.session_state:
                st.warning("Previous query discarded.")
                st.session_state.user_question = ""
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done ✅")
            else:
                st.warning("Please upload PDF files before processing.")

    # ===== User Question =====
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""

    # ===== MCQ & SAQ Generation =====
    if pdf_docs and api_key:
        text = get_pdf_text(pdf_docs)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📘 Generate MCQs", key="mcq_btn"):
                with st.spinner("Generating MCQs..."):
                    mcqs = mcq_chain_function(llm, text)
                    st.success("MCQs Generated ✅")
                    st.markdown(f"### 📘 Multiple Choice Questions\n\n{mcqs}")

        with col2:
            if st.button("✍️ Generate SAQs", key="saq_btn"):
                with st.spinner("Generating SAQs..."):
                    saqs = saq_chain_function(llm, text)
                    st.success("SAQs Generated ✅")
                    st.markdown(f"### ✍️ Short Answer Questions\n\n{saqs}")


if __name__ == "__main__":
    main()
