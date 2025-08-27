from datetime import datetime
import html
import streamlit as st
import os
from dotenv import load_dotenv

from src.loaders import create_document_loader
from src.embeddings import get_embedding_manager
from src.vectorstore import get_vectorstore_manager
from src.chain import create_conversational_chain

load_dotenv()

st.set_page_config(page_title="RagBot", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

def ts_now() -> str:
    now = datetime.now()
    hour = now.hour % 12 or 12
    minute = f"{now.minute:02d}"
    ampm = "AM" if now.hour < 12 else "PM"
    return f"{hour}:{minute} {ampm}"

def safe_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")

def get_bot_reply(history: list[dict], prompt: str) -> str:
    if st.session_state.rag_chain and st.session_state.vectorstore_ready:
        try:
            response = st.session_state.rag_chain.get_response(prompt)
            return response["answer"]
        except Exception as e:
            st.error(f"Error with RAG chain: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    else:
        return "Please upload documents first to enable AI-powered responses based on your content."

def setup_rag_system():
    try:
        embedding_manager = get_embedding_manager()
        embeddings = embedding_manager.embeddings
        
        vectorstore_manager = get_vectorstore_manager(embeddings)
        
        if vectorstore_manager.load_vectorstore(show_messages=False):
            st.session_state.vectorstore_ready = True
            retriever = vectorstore_manager.get_retriever()
            st.session_state.rag_chain = create_conversational_chain(retriever)
            st.success("RAG system loaded from saved vector store!")
        
        return embedding_manager, vectorstore_manager
        
    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None, None

with st.sidebar:
    st.title("Settings")
    
    embedding_manager, vectorstore_manager = setup_rag_system()
    
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to enable AI-powered responses"
    )
    
    if uploaded_files and embedding_manager and vectorstore_manager:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                doc_loader = create_document_loader()
                chunks = doc_loader.process_uploaded_files(uploaded_files)
                
                if chunks:
                    if not st.session_state.vectorstore_ready:
                        vectorstore_manager.create_vectorstore(chunks)
                        st.session_state.vectorstore_ready = True
                    else:
                        vectorstore_manager.add_documents(chunks)
                    
                    vectorstore_manager.save_vectorstore()
                    
                    retriever = vectorstore_manager.get_retriever()
                    st.session_state.rag_chain = create_conversational_chain(retriever)
                    
                    st.success("Documents processed successfully!")
                    st.rerun()
    
    st.subheader("System Status")
    
    vector_files_exist = os.path.exists("vectorstore/faiss_index.faiss")
    
    if st.session_state.vectorstore_ready:
        if vector_files_exist:
            st.success("Vector store ready (loaded from disk)")
        else:
            st.success("Vector store ready (created this session)")
    else:
        if vector_files_exist:
            st.info("Saved vector store available - restart to load")
        else:
            st.warning("No documents loaded - upload PDFs to get started")
    
    if st.session_state.rag_chain:
        st.success("RAG chain active")
    else:
        st.warning("RAG chain not initialized")
    
    st.subheader("Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat"):
            st.session_state.messages.clear()
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.rerun()
    

st.markdown("""
<style>
  body, .main, .block-container { background: #0f1116; }
  .block-container { max-width: 900px; }
  h1, h2, h3, .stMarkdown, .stTextInput, .stChatInput { color: #f4f4f5 !important; }

  .chat-row { display: flex; margin: 10px 0; align-items: flex-end; gap: 8px; }
  .chat-row.right { justify-content: flex-end; }
  .chat-row.left  { justify-content: flex-start; }

  .bubble { padding: 10px 14px; border-radius: 12px; max-width: 70%; line-height: 1.4; }
  .bubble.user { background: #DCF8C6; color: #111; }
  .bubble.bot  { background: #F1F0F0; color: #111; }

  .avatar { font-size: 28px; }
  .timestamp { font-size: .8rem; opacity: .6; margin: 2px 6px; }
  .timestamp.right { text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("RagBot")

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-row right">
              <div class="bubble user">{safe_html(m["content"])}</div>
              <div class="avatar">🧑</div>
            </div>
            <div class="timestamp right">{m["ts"]}</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-row left">
              <div class="avatar">🤖</div>
              <div class="bubble bot">{safe_html(m["content"])}</div>
            </div>
            <div class="timestamp">{m["ts"]}</div>
            """,
            unsafe_allow_html=True
        )

prompt = st.chat_input("Pass your prompt here!")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": ts_now()})

    reply = get_bot_reply(st.session_state.messages[:-1], prompt)

    st.session_state.messages.append({"role": "assistant", "content": reply, "ts": ts_now()})

    st.rerun()

