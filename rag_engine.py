from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

from typing import Optional, List
import requests
from pydantic import Field



# OpenRouter LLM 

class OpenRouterLLM(LLM):
    api_key: str = Field(..., description="Enter your api key")
    model: str = Field(
        default="mistralai/mistral-7b-instruct:free",
        description="OpenRouter model name"
    )

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "RAG-Document-QA"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.2
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.text}")

        return response.json()["choices"][0]["message"]["content"]



# Main RAG Engine

class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.vectorstore = None
        self.qa_chain = None

        # Embedding model (local, free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # LLM
        self.llm = OpenRouterLLM(api_key=api_key)

        # Prompt to prevent hallucinations
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use ONLY the context below to answer the question.\n"
                "If the answer is not present in the context, say "
                "'Not found in the document.'\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}"
            )
        )

    
    # Document Processing

    def process_document(
        self,
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    
    # Query Interface
   
    def query(self, question: str) -> dict:
        if not self.qa_chain:
            raise ValueError("Document not processed yet.")

        result = self.qa_chain({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page", None)
            preview = doc.page_content[:120].replace("\n", " ")
            if page is not None:
                sources.append(f"Page {page + 1}: {preview}...")
            else:
                sources.append(preview + "...")

        return {
            "answer": result["result"],
            "sources": sources
        }

    
    # Optional Persistence
    
    def save_index(self, path: str = "faiss_index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_index(self, path: str = "faiss_index"):
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

        