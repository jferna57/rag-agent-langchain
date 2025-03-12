import os
import logging
import socket
import time
from typing import List, Dict, Callable
from functools import wraps


from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.data import save_data, DataPayload, SystemInfo, ModelInfo
from src.ingestion import load_pdf
from src.chunking import split_text
from src.vector_db import setup_vector_db
from src.model_loader import load_llm
from src.retrieval import setup_retriever
from src.prompt_template import get_query_prompt
from src.utils import obtener_info_equipo

# Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global dictionary to store step times
performance_data: Dict[str, float] = {}

def timed_function(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function and store it in steps_times.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: The wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time.time()
        result = func(*args, **kwargs)
        end_time: float = time.time()
        elapsed_time: float = end_time - start_time
        logging.info('Function %s took %4f seconds to execute.', func.__name__, elapsed_time)
        
        # Store the elapsed time in the global dictionary
        performance_data[func.__name__] = elapsed_time

        return result

    return wrapper

@timed_function
def load_and_split_pdf(pdf_file: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Loads a PDF and splits it into chunks."""
    logging.info("Loading and splitting PDF...")
    documents = load_pdf(pdf_file)

    if not documents:
        logging.error("Error loading PDF.")
        return None,0

    chunks = split_text (documents, chunk_size, chunk_overlap)
    if not chunks:
        logging.error("Error splitting text.")
        return None,0
    logging.info("Text split into %s chunks", len(chunks))
    
    return chunks

@timed_function
def setup_vector_database(chunks: List, embedding_model: str, collection_name: str) -> Chroma:
    """Sets up the vector database."""
    logging.info("Setting up vector database...")
    vector_db = setup_vector_db (chunks, embedding_model, collection_name)
    if not vector_db:
        logging.error("Error setting up vector database.")
        return None
    return vector_db

@timed_function
def load_language_model(model_name: str):
    """Loads the language model."""
    logging.info("Loading language model...")
    llm = load_llm(model_name)
    if not llm:
        logging.error("Error loading language model: %s", model_name)
        return None
    return llm

@timed_function
def setup_retrieval_system(vector_db, llm):
    """Sets up the retrieval system."""
    logging.info("Setting up retrieval system...")
    query_prompt = get_query_prompt()
    retriever = setup_retriever(vector_db, llm, query_prompt)
    if not retriever:
        logging.error("Error setting up retrieval system.")
        return None
    return retriever

@timed_function
def execute_llm_query(retriever, llm, question: str) -> Dict:
    """Executes a query and returns the result and elapsed time."""
    logging.info('Executing query: %s', question)
    template = ChatPromptTemplate.from_template("Answer the question based ONLY on the following context: {context}\nQuestion: {question}")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(input=(question,))
    return {"question": question, "answer": result}

def main():
    """Main function to execute the processing pipeline."""
    system_info: SystemInfo = obtener_info_equipo()

    # Load environment variables
    load_dotenv(override=True)

    pdf_file = os.getenv("PDF_FILE")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    collection_name = os.getenv("COLLECTION_NAME")
    model_name = os.getenv("MODEL_NAME")

    # Print environment variables
    logging.info("PDF_FILE: %s", pdf_file)
    logging.info("EMBEDDING_MODEL: %s", embedding_model)
    logging.info("COLLECTION_NAME: %s", collection_name)
    logging.info("MODEL_NAME: %s", model_name)

    questions_and_answers: List[Dict] = []

    # 1. Load and split PDF
    chunks = load_and_split_pdf(pdf_file, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    # 2. Setup vector database
    vector_db = setup_vector_database(chunks, embedding_model, collection_name)
    if vector_db is None:
        return

    # 3. Load language model
    llm = load_language_model(model_name)
    if llm is None:
        return

    # 4. Setup retrieval system
    retriever = setup_retrieval_system(vector_db, llm)
    if retriever is None:
        return

    # 5. Execute queries
    query_results = [
        execute_llm_query(retriever, llm, "Genera un resumen del documento")
        # execute_query(retriever, llm, "Dime el titulo del documento")
    ]

    # Store the questions and answers
    questions_and_answers.append({
        "question": query_results[0]["question"], 
        "answer": query_results[0]["answer"]
    })
    # questions_and_answers.append({"question": query_results[1]["question"], "answer": query_results[1]["answer"]})


    # Create ModelInfo object
    model_info_obj = ModelInfo(
        model_name=model_name,
        embedding_model=embedding_model
    )

    # Create PerformanceData object
    performance_data_obj = performance_data

    # Create DataPayload object
    data_payload = DataPayload(
        server_name=socket.gethostname(),
        timestamp="",
        server_data=system_info,
        performance_data=performance_data_obj,
        model_info=model_info_obj,
        questions_and_answers=questions_and_answers
    )
    save_data(data_payload)

if __name__ == '__main__':
    main()
