import os
import tempfile
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, uploaded_files) -> List[Document]:
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Load PDF
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["file_type"] = "pdf"
                
                all_documents.extend(documents)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
    
    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        if not uploaded_files:
            return []
        
        documents = self.load_documents(uploaded_files)
        
        if not documents:
            return []
        
        chunks = self.split_documents(documents)
        
        st.success(f"Loaded {len(uploaded_files)} files")
        # st.success(f"Processed {len(uploaded_files)} files into {len(chunks)} chunks")
        
        return chunks


def create_document_loader(chunk_size: int = 1000, chunk_overlap: int = 200) -> DocumentLoader:
    return DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
