import os
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import fitz
from tqdm import tqdm

# Set embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Set vector store directory
CHROMA_PATH = "risk_index_db"

# 1. Load and preprocess document
def load_annual_report(file_path):
    print("[INFO] Loading and processing document...")
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# 2. Split into chunks
def split_text(text, chunk_size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# 3. Embed and store in ChromaDB
def index_chunks(chunks):
    print("[INFO] Creating vector database...")
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model, persist_directory=CHROMA_PATH)
    vectordb.persist()
    return vectordb

# 4. Retrieve relevant chunks
def retrieve(query, vectordb, k=5):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

# 5. Prompt Ollama LLM for summarization and risk scoring
def generate_risk_summary_and_score(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are a financial analyst AI agent.

Given the following excerpts from an annual report, identify and summarize the main risks associated with the company.
Then, assign a risk score from 0 (no risk) to 10 (extremely high risk).

Context:
{context}

User Query:
{query}

Respond in this format:
1- use the context to extarct key risk related information from the document
2- Summarize the main risks identified, with an explanation on why they cna be considered risks:
 Risk 1
 Risk 2
...
3- give the overall risk score , Risk Score: <number between 0 and 10>, and also provide a detailed explantion for the risk score.

"""
    print("[INFO] Querying LLM for risk analysis...")
    response = ollama.chat(
        model='llama3',  # or whichever model you pulled
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# 6. Main flow
def main():
    file_path = input("Enter path to the annual report (PDF/DOC/TXT): ").strip()
    user_query = input("Enter your query (e.g., 'Identify risks and assign a score'): ").strip()

    raw_text = load_annual_report(file_path)
    chunks = split_text(raw_text)
    vectordb = index_chunks(chunks)
    retrieved_docs = retrieve(user_query, vectordb)
    response = generate_risk_summary_and_score(user_query, retrieved_docs)

    print("\n==== AI Agent Response ====\n")
    print(response)

if __name__ == "__main__":
    main()
