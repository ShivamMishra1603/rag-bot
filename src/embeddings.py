from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import streamlit as st


class EmbeddingManager:
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self._embeddings = None
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},     
                    encode_kwargs={'normalize_embeddings': True} 
                )
                st.success(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                st.error(f"Failed to load embedding model: {str(e)}")
                raise e
        
        return self._embeddings


@st.cache_resource
def get_embedding_manager(model_name: str = "BAAI/bge-base-en-v1.5") -> EmbeddingManager:
    return EmbeddingManager(model_name=model_name)
