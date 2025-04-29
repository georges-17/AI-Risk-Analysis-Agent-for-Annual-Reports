# AI-Risk-Analysis-Agent-for-Annual-Reports
his project is an AI-powered agent that analyzes company annual reports (PDF/DOC/TXT) to identify and summarize key risks, then assigns a risk score from 0 to 10. It uses local LLMs (via Ollama), PDF parsing with fitz, text embedding with Sentence Transformers, and vector similarity search with  ChromaDB.
🔍 Key Features
📄 PDF Parsing: Extracts and processes text from annual reports using fitz (PyMuPDF)

🧩 Chunking & Embedding: Splits documents into manageable chunks and embeds them using all-MiniLM-L6-v2

🔎 Vector Search: Retrieves the most relevant document sections via vector similarity (ChromaDB or FAISS)

🤖 Local LLM Integration: Uses ollama to generate risk summaries and scores using models like llama3

📈 Risk Scoring: Returns structured output with identified risks, explanations, and an overall risk score

💻 Tech Stack
Python

Ollama (Llama 3 or other models)

PyMuPDF (fitz)

LangChain

Sentence Transformers

ChromaDB

📦 How to Use
Clone the repo

Install dependencies (see req.txt)

Run the agent and provide a PDF/DOC/TXT file

Enter your query (e.g., "Identify risks and assign a score")

Get an AI-generated risk analysis

