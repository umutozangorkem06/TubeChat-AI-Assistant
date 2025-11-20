"""
RAG Engine for TubeChat application.

This module handles:
- Text embeddings using OpenAI Embeddings
- FAISS vector store creation and management
- Question-answering chain setup using LangChain
- Context-aware responses based on video transcripts
"""

from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for processing video transcripts
    and answering questions based on video content.
    
    This class manages the entire RAG pipeline:
    1. Text chunking for efficient processing
    2. Embedding generation using OpenAI
    3. Vector store creation with FAISS
    4. Question-answering using LangChain's RetrievalQA chain
    
    Attributes:
        api_key (str): OpenAI API key
        embeddings (OpenAIEmbeddings): Embedding model instance
        llm (ChatOpenAI): Language model instance (GPT-3.5-turbo)
        vector_store (Optional[FAISS]): FAISS vector store for similarity search
        qa_chain (Optional[RetrievalQA]): QA chain for answering questions
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the RAG Engine.
        
        Args:
            api_key: OpenAI API key for embeddings and LLM
            model_name: Name of the OpenAI model to use. Default is "gpt-3.5-turbo"
            temperature: Temperature for LLM responses (0.0-2.0). 
                        Higher values make output more random. Default is 0.7.
                        
        Raises:
            ValueError: If API key is empty or invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("OpenAI API key is required and must be a non-empty string.")
        
        self.api_key = api_key
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-ada-002"  # OpenAI's embedding model
        )
        
        # Initialize OpenAI chat model
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Vector store and QA chain will be initialized when text is processed
        self.vector_store: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None
        
    def create_vector_store(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """
        Create a FAISS vector store from the provided text.
        
        This method:
        1. Splits the text into chunks using RecursiveCharacterTextSplitter
        2. Generates embeddings for each chunk
        3. Stores embeddings in a FAISS vector database
        4. Creates a QA chain for question answering
        
        Args:
            text: The text to embed and store (typically video transcript)
            chunk_size: Maximum size of each text chunk. Default is 1000 characters.
            chunk_overlap: Number of characters to overlap between chunks. 
                          Default is 200 characters.
                          
        Raises:
            ValueError: If text is empty or invalid
            Exception: If embedding or vector store creation fails
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string.")
        
        if not text.strip():
            raise ValueError("Text cannot be empty or whitespace only.")
        
        # Use RecursiveCharacterTextSplitter for intelligent chunking
        # This splitter tries to keep related text together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split hierarchy
        )
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        
        if not chunks:
            raise ValueError("Text splitting resulted in no chunks. Please check the input text.")
        
        # Create vector store from chunks using OpenAI embeddings
        try:
            self.vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )
        except Exception as e:
            error_msg = str(e)
            # Handle specific OpenAI API errors
            if "insufficient_quota" in error_msg.lower() or "429" in error_msg:
                raise ValueError(
                    "OpenAI API quota exceeded. Please check your account balance and billing details. "
                    "Visit https://platform.openai.com/account/billing to add credits or upgrade your plan."
                )
            elif "rate limit" in error_msg.lower():
                raise ValueError(
                    "OpenAI API rate limit exceeded. Please wait a moment and try again. "
                    "If this persists, check your API usage limits at https://platform.openai.com/account/usage"
                )
            elif "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise ValueError(
                    "Invalid OpenAI API key. Please check your API key in the configuration. "
                    "Get a valid key from https://platform.openai.com/api-keys"
                )
            else:
                raise ValueError(
                    f"Failed to create embeddings: {error_msg}. "
                    "Please check your OpenAI API key and account status."
                )
        
        # Create QA chain with custom prompt
        self._create_qa_chain()
    
    def _create_qa_chain(self) -> None:
        """
        Create a RetrievalQA chain with a custom prompt template.
        
        The chain uses:
        - The initialized LLM (GPT-3.5-turbo)
        - The FAISS vector store as a retriever
        - A custom prompt template for context-aware responses
        - Top 4 most relevant chunks for context
        
        This is a private method called automatically after vector store creation.
        """
        # Custom prompt template for better context-aware responses
        # The template instructs the model to:
        # 1. Use only the provided context
        # 2. Admit when it doesn't know the answer
        # 3. Be concise and accurate
        prompt_template = """You are a helpful assistant that answers questions based on the context provided from a YouTube video transcript.

Use the following pieces of context from the video transcript to answer the question.
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
Be concise, accurate, and focus on the information from the video transcript.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        # "stuff" chain type puts all retrieved documents into the prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple chain that stuffs all docs into prompt
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False  # Don't return source docs in response
        )
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        
        This method:
        1. Validates that the vector store is initialized
        2. Retrieves relevant context chunks from the vector store
        3. Generates an answer using the LLM with the retrieved context
        4. Returns the answer as a string
        
        Args:
            question: The user's question about the video content
            
        Returns:
            The answer based on the video transcript context
            
        Raises:
            ValueError: If vector store is not initialized
            Exception: If query processing fails
        """
        if self.qa_chain is None or self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Please process a video transcript first "
                "by calling create_vector_store() before querying."
            )
        
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        
        try:
            # Invoke the QA chain with the question
            result = self.qa_chain.invoke({"query": question})
            
            # Extract the answer from the result
            answer = result.get("result", "")
            
            if not answer:
                return "I couldn't generate an answer. Please try rephrasing your question."
            
            return answer
            
        except Exception as e:
            # Return a user-friendly error message
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                return (
                    "OpenAI API rate limit exceeded. Please wait a moment and try again. "
                    "If this persists, check your API usage limits."
                )
            elif "insufficient_quota" in error_msg.lower():
                return (
                    "OpenAI API quota exceeded. Please check your API account balance "
                    "and billing information."
                )
            elif "invalid_api_key" in error_msg.lower():
                return (
                    "Invalid OpenAI API key. Please check your API key in the configuration."
                )
            else:
                return f"An error occurred while processing your question: {error_msg}. Please try again."
    
    def is_initialized(self) -> bool:
        """
        Check if the RAG engine is initialized with a vector store.
        
        Returns:
            True if vector store exists and is ready for queries, False otherwise
        """
        return self.vector_store is not None and self.qa_chain is not None
    
    def get_vector_store_info(self) -> dict:
        """
        Get information about the current vector store.
        
        Returns:
            Dictionary with vector store information including:
            - is_initialized: Whether vector store exists
            - num_documents: Number of documents in the vector store (if initialized)
            
        Note:
            This is a helper method for debugging and monitoring.
        """
        info = {
            "is_initialized": self.is_initialized()
        }
        
        if self.vector_store is not None:
            # FAISS doesn't directly expose document count, but we can check if it exists
            info["vector_store_exists"] = True
        else:
            info["vector_store_exists"] = False
        
        return info
