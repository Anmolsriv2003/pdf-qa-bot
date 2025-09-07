import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = "data"

def list_pdfs():
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]
    return files

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Force CPU to avoid meta tensor error
    )
    return FAISS.from_documents(docs, embeddings)

def ask_question(vector_db, query):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant" 
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        return_source_documents=False
    )
    return qa.invoke({"query": query})["result"]

if __name__ == "__main__":
    pdfs = list_pdfs()
    if not pdfs:
        print("No PDF files found in data folder.")
        exit()

    print("Available PDFs:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"{i}. {pdf}")

    choice = int(input("Select a PDF by number: ")) - 1
    selected_pdf = os.path.join(DATA_FOLDER, pdfs[choice])

    print(f"\nLoading {pdfs[choice]} ...")
    documents = load_pdf(selected_pdf)
    docs = split_documents(documents)
    vector_db = create_vector_db(docs)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_question(vector_db, query)
        print("\nAnswer:", answer)
