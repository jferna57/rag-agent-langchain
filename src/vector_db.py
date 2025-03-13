import logging
from typing import List

import ollama
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def setup_vector_db(chunks: List[Document], embedding_model: str):
    """
    Configures a vector database using FAISS and OllamaEmbeddings, storing it in memory.

    This function creates a new FAISS vector database in memory.

    Args:
        chunks (List[Document]): List of document chunks.
        embedding_model (str): Name of the Ollama model for generating embeddings.
        collection_name (str): Name of the collection in the vector database.

    Returns:
        FAISS: Instance of the configured vector database.
        None: If an error occurs during configuration.
    """
    try:
        # Download the embedding model from Ollama
        ollama.pull(embedding_model)

        # Create embeddings using Ollama
        embeddings = OllamaEmbeddings(model=embedding_model)

        # Create a FAISS vector store from the document chunks
        vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)

        logging.info("FAISS vector database configured correctly (in memory)")
        return vector_db

    except Exception as e:
        logging.error("Error configuring the FAISS vector database: %s", e)
        return None
