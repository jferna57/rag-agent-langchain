import logging
from typing import List

import chromadb
import ollama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def setup_vector_db(chunks: List[Document], embedding_model: str, collection_name: str):
    """
    Configures a vector database using Chroma and OllamaEmbeddings, storing it in memory.

    This function creates a new vector database in memory.

    Args:
        chunks (List[Document]): List of document chunks.
        embedding_model (str): Name of the Ollama model for generating embeddings.
        collection_name (str): Name of the collection in the vector database.

    Returns:
        Chroma: Instance of the configured vector database.
        None: If an error occurs during configuration.
    """
    try:
        # Download the embedding model from Ollama
        ollama.pull(embedding_model)

        # disable chromadb telemetry
        chromadb_config_settings = chromadb.config.Settings(
            is_persistent=False,
            anonymized_telemetry=False,
        )
        # Create a new database in memory and add the documents
        vector_db = Chroma.from_documents(
            client_settings=chromadb_config_settings,
            documents=chunks,
            embedding=OllamaEmbeddings(model=embedding_model),
            collection_name=collection_name,
            persist_directory=None  # No persistent directory
        )
        logging.info("Vector database configured correctly (in memory)")
        return vector_db
    except Exception as e:
        logging.error(f"Error configuring the vector database: {e}")
        return None
