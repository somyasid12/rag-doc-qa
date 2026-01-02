RAG Document Q&A System
A lightweight Retrieval-Augmented Generation (RAG) application that lets users upload a PDF and ask questions strictly based on its content.
The system retrieves relevant parts of the document and uses an LLM to generate grounded answers, reducing hallucinations.
This project was built as a practical implementation of RAG, focusing on correctness, clarity, and real-world structure rather than overengineering.
✨ Features
Upload and process PDF documents
Semantic search using vector embeddings
Question answering grounded only in document content
Source snippets shown for transparency
Simple, clean Streamlit interface
🧠 How it works (High Level)
The uploaded PDF is split into text chunks
Each chunk is converted into embeddings using Sentence Transformers
Embeddings are stored in a FAISS vector index
For a user question, relevant chunks are retrieved
The retrieved context is passed to an LLM via OpenRouter
The model answers only using the retrieved content
This retrieval step ensures the model does not answer from memory.
🛠️ Tech Stack
Python 3.11
Streamlit – UI
LangChain – RAG pipeline
FAISS – Vector database
Sentence Transformers – Embeddings
OpenRouter API – LLM access
Mistral 7B Instruct (free) – Generation model
📁 Project Structure
rag-doc-qa/
│
├── app.py              # Streamlit UI
├── rag_engine.py       # Document ingestion + retrieval + generation
├── requirements.txt    # Dependencies
├── README.md
└── .gitignore
⚙️ Setup & Run
1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
2. Install dependencies
pip install -r requirements.txt
3. Run the app
python -m streamlit run app.py
Open in browser:
http://localhost:8501
🚀 Usage
Enter your OpenRouter API key
Upload a PDF document
Click Process Document
Ask questions in the chat interface
Example questions:
What is this document about?
Summarize the key points
What does it say about surcharge rates?
📌 Why RAG?
Large language models can hallucinate when answering from memory.
RAG avoids this by:
Retrieving relevant context from a trusted source
Passing only that context to the model
Forcing answers to be grounded in the document
This approach is useful for:
Policy documents
Academic notes
Government / legal material
Internal knowledge bases
