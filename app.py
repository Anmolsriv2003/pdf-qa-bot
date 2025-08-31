import streamlit as st
import os
from main import create_vector_db, load_pdf, split_documents, ask_question, DATA_FOLDER, list_pdfs

st.set_page_config(page_title="📄 PDF-QA Bot", layout="wide", page_icon="📄")

st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #0a0a0a;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 8px;
        border: none;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        height: 2.5em;
        font-size: 16px;
    }
    .stSelectbox>div>div>div>select {
        border-radius: 8px;
        height: 2.5em;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📄 PDF-QA Bot")
st.markdown("Ask questions from your PDFs and get instant answers!")

with st.sidebar:
    st.header("Upload or Select PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    pdfs = list_pdfs()
    
    if uploaded_file:
        os.makedirs(DATA_FOLDER, exist_ok=True)
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        selected_pdf = file_path
    elif pdfs:
        selected_pdf_name = st.selectbox("Select a PDF from folder", pdfs)
        selected_pdf = os.path.join(DATA_FOLDER, selected_pdf_name)
    else:
        st.info("No PDF available. Upload one to start.")
        st.stop()

with st.spinner(f"Loading {os.path.basename(selected_pdf)} ..."):
    documents = load_pdf(selected_pdf)
    docs = split_documents(documents)
    vector_db = create_vector_db(docs)
st.success(f"{os.path.basename(selected_pdf)} loaded successfully!")

st.subheader("Ask a question about the PDF:")
query = st.text_input("Type your question here:")

if st.button("Get Answer") and query:
    with st.spinner("Generating answer..."):
        answer = ask_question(vector_db, query)
    st.markdown("### 💡 Answer")
    st.markdown(f"<div style='background-color:#e6f7ff;padding:15px;border-radius:8px'>{answer}</div>", unsafe_allow_html=True)
