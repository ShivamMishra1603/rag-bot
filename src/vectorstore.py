import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import pickle


class VectorStoreManager:
    
    def __init__(self, embeddings: HuggingFaceEmbeddings, persist_directory: str = "vectorstore"):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore: Optional[FAISS] = None
        
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        try:
            with st.spinner("Creating vector store..."):
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            st.success(f"Created vector store with {len(documents)} documents")
            return self.vectorstore
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            raise e
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            st.warning("No documents to add")
            return
        
        if self.vectorstore is None:
            st.warning("No existing vector store. Creating new one...")
            self.create_vectorstore(documents)
            return
        
        try:
            with st.spinner("Adding documents to vector store..."):
                self.vectorstore.add_documents(documents)
            st.success(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            raise e
    
    def save_vectorstore(self, name: str = "faiss_index") -> None:
        if self.vectorstore is None:
            st.warning("No vector store to save")
            return
        
        try:
            save_path = os.path.join(self.persist_directory, name)
            self.vectorstore.save_local(save_path)
            st.success(f"Vector store saved to {save_path}")
            
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
            raise e
    
    def load_vectorstore(self, name: str = "faiss_index", show_messages: bool = True) -> Optional[FAISS]:
        try:
            load_path = os.path.join(self.persist_directory, name)
            
            if not os.path.exists(f"{load_path}.faiss"):
                if show_messages:
                    st.info(f"No saved vector store found at {load_path}")
                return None
            
            with st.spinner("Loading vector store..."):
                self.vectorstore = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            if show_messages:
                st.success("Vector store loaded successfully")
            return self.vectorstore
            
        except Exception as e:
            if show_messages:
                st.error(f"Error loading vector store: {str(e)}")
            return None
    
    def get_retriever(self, search_type: str = "similarity", k: int = 4, **kwargs):

        if self.vectorstore is None:
            raise ValueError("No vector store available. Please create or load one first.")
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        
        if self.vectorstore is None:
            raise ValueError("No vector store available")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def reset_vectorstore(self) -> None:
        self.vectorstore = None
        st.info("Vector store reset")


@st.cache_resource
def get_vectorstore_manager(_embeddings: HuggingFaceEmbeddings, persist_directory: str = "vectorstore") -> VectorStoreManager:
    
    return VectorStoreManager(embeddings=_embeddings, persist_directory=persist_directory)
