import streamlit as st
import os
from main import list_pdfs, load_pdf, split_documents, create_vector_db, ask_question

st.set_page_config(
    page_title="📄 PDF-QA Bot",
    layout="wide",
    page_icon="📚"
)

st.markdown(
    """
    <style>
    /* Gradient background */
    body, .stApp {
        background: linear-gradient(135deg, #f5f7fa, #e0f7fa);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title and subtitle */
    h1, h2, h3 {
        color: #0d47a1;
    }

    /* Sidebar */
    .css-1d391kg .stSelectbox>div>div>div {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 5px;
    }

    /* Button */
    .stButton>button {
        background-color: #0d47a1;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1976d2;
        transform: scale(1.05);
    }

    /* Text input */
    .stTextInput>div>input {
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #90caf9;
        width: 100%;
    }

    /* Answer box with scroll */
    .answer-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.15);
        font-size: 16px;
        line-height: 1.5;
        max-height: 300px;
        overflow-y: auto;
    }

    /* Question box */
    .question-box {
        background-color: #e3f2fd;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        font-weight: bold;
    }

    /* Footer */
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #555555;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("## ℹ️ Instructions")
st.sidebar.markdown(
    """
    - Select a PDF from the dropdown.  
    - Type your question in the input box.  
    - Get instant answers powered by LLMs + LangChain.  
    - Scroll answers if they are long.  
    - Use emojis 😄📚 to make it fun!
    """
)

st.title("📄 PDF-QA Bot 🤖")
st.write("Ask questions from your PDFs and get instant answers! 📚✨")

pdfs = list_pdfs()
if not pdfs:
    st.warning("No PDF files found in `data/` folder.")
    st.stop()

selected_pdf_name = st.sidebar.selectbox("Select a PDF 📄", pdfs)
selected_pdf = os.path.join("data", selected_pdf_name)

with st.spinner(f"Loading {selected_pdf_name} ... ⏳"):
    documents = load_pdf(selected_pdf)
    docs = split_documents(documents)
    vector_db = create_vector_db(docs)

query = st.text_input("💬 Enter your question:")


if query:
    st.markdown(f"<div class='question-box'>❓ {query}</div>", unsafe_allow_html=True)
    with st.spinner("Fetching answer ... 🤖"):
        answer = ask_question(vector_db, query)
        st.markdown(f"<div class='answer-box'>✅ {answer}</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>🚀 Powered by LLM + LangChain + Streamlit</div>",
    unsafe_allow_html=True
)
