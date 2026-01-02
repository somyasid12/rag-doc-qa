RAG Document Q&A System
This project is a simple Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions based strictly on its content.
The system retrieves relevant sections from the document and uses an LLM to generate grounded answers, reducing hallucinations.
The project was built as a hands-on implementation of RAG, focusing on correctness, explainability, and clean structure rather than overengineering.
🔍 What this project does
Upload a PDF document
Split and embed document content using Sentence Transformers
Store embeddings in FAISS
Retrieve relevant chunks for a user query
Generate answers using an LLM via OpenRouter
Show source snippets used to answer the question
Answers are generated only from the document content.
🧠 Why RAG?
Large language models can hallucinate when answering from memory.
RAG solves this by:
Retrieving relevant context from a trusted source (the document)
Passing only that context to the LLM
Forcing the model to answer based on retrieved text
This approach is especially useful for:
Policy documents
Study material
Government / legal documents
Internal knowledge bases
🏗️ Tech Stack
Python 3.11
Streamlit – UI
LangChain – RAG pipeline
FAISS – Vector database
Sentence Transformers – Embeddings
OpenRouter API – LLM access
Mistral 7B Instruct (free) – Generation model
