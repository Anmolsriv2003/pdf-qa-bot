import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

DATA_FOLDER = "data"

# ----------------------
# Prompt Template with context
# ----------------------
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an expert assistant. Use the context below from the documents to answer the user's question clearly and concisely.

Context:
{context}

Question: {question}

Answer:
"""
)

# ----------------------
# PDF utilities
# ----------------------
def list_pdfs():
    return [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# ----------------------
# Ask question with context
# ----------------------
def ask_question(vector_db, query, top_k=3):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )

    # Retrieve top_k relevant document chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)
    
    # Combine text from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format query with prompt template
    prompt_text = prompt_template.format(question=query, context=context)

    # ✅ Return output string directly
    return llm.invoke(prompt_text).content

# ----------------------
# Main program
# ----------------------
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
