import os
from typing import Dict, List, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.schema import BaseRetriever, BaseMessage
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class ConversationalRAGChain:
    def __init__(
        self,
        retriever: BaseRetriever,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        memory_window: int = 10
    ):
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.memory_window = memory_window
        
        self.llm = self._create_llm()
        self.memory = self._create_memory()
        self.chain = self._create_chain()
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=api_key
        )
    
    def _create_memory(self) -> ConversationBufferWindowMemory:
        # Using the current implementation but with proper message history
        chat_history = ChatMessageHistory()
        return ConversationBufferWindowMemory(
            k=self.memory_window,
            memory_key="chat_history",
            output_key="answer",
            return_messages=True,
            chat_memory=chat_history
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        custom_template = """You are a helpful AI assistant that answers questions based on the provided context from uploaded documents. 

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't have enough information to answer the question. Don't try to make up an answer.

Always be conversational and helpful. If the context provides relevant information, use it to give a comprehensive answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=custom_template,
                    input_variables=["context", "chat_history", "question"]
                )
            }
        )
    
    def get_response(self, question: str) -> Dict[str, Any]:
        try:
            with st.spinner("Thinking..."):
                result = self.chain.invoke({"question": question})
            
            return {
                "answer": result.get("answer", "I couldn't generate a response."),
                "source_documents": result.get("source_documents", [])
            }
            
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.",
                "source_documents": []
            }
    
    def clear_memory(self) -> None:
        self.memory.clear()
        st.info("Conversation memory cleared")
    
    def get_memory_summary(self) -> str:
        return str(self.memory.buffer)


def create_conversational_chain(
    retriever: BaseRetriever,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    memory_window: int = 10
) -> ConversationalRAGChain:
    return ConversationalRAGChain(
        retriever=retriever,
        model_name=model_name,
        temperature=temperature,
        memory_window=memory_window
    )
