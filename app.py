import streamlit as st
import os
import uuid
from rag_engine import RAGEngine


# Page Configuration

st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide"
)


# Session State Initialization

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False



# Title

st.title("📚 RAG Document Q&A")
st.markdown(
    "Upload a document and ask questions. "
    "Answers are generated **only from the document content**."
)


# Sidebar

with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Get your API key from https://openrouter.ai"
    )

    st.divider()

    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded_file and api_key:
        if st.button("🔄 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                try:
                    # Save file temporarily
                    temp_path = f"temp_{uuid.uuid4()}.pdf"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Initialize RAG engine
                    st.session_state.rag_engine = RAGEngine(api_key)
                    st.session_state.rag_engine.process_document(temp_path)

                    st.session_state.indexed = True
                    st.session_state.messages = []

                    # Cleanup temp file
                    os.remove(temp_path)

                    st.success("✅ Document indexed successfully!")

                except Exception as e:
                    st.error(f"❌ Error while processing document: {e}")

    if st.session_state.indexed:
        st.success("🟢 System Ready")

        st.divider()

        st.subheader("🔧 System Info")
        st.markdown(
            """
            - **Vector Store:** FAISS  
            - **Embeddings:** Sentence Transformers  
            - **LLM:** OpenRouter  
            - **Framework:** LangChain  
            """
        )

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()



# Main Layout

col1, col2 = st.columns([3, 1])

# Chat Column

with col1:
    st.header("💬 Chat")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- {src}")

    
    
# Chat Input 

if st.session_state.indexed:
    prompt = st.chat_input("Ask a question about the document...")
else:
    prompt = None

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_engine.query(prompt)
            st.markdown(response["answer"])

            if response["sources"]:
                with st.expander("📎 Sources"):
                    for src in response["sources"]:
                        st.markdown(f"- {src}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"]
            })



# Instructions Column
with col2:
    st.header("ℹ️ How to Use")
    st.markdown(
        """
        **Steps**
        1. Enter your OpenRouter API key  
        2. Upload a PDF document  
        3. Click **Process Document**  
        4. Ask questions in the chat  

        **Example Questions**
        - What is this document about?
        - Summarize the key points
        - What does it say about a specific topic?
        """
    )

    if not st.session_state.indexed:
        st.info("👆 Upload a document to begin")
