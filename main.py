import dataclasses
import logging
import os
import socket
import sys
import time
from functools import wraps
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.chunking import split_text
from src.data import DataPayload, ModelInfo, SystemInfo, save_data
from src.ingestion import load_pdf
from src.model_loader import load_llm
from src.prompt_template import get_query_prompt
from src.retrieval import setup_retriever
from src.utils import obtener_info_equipo
from src.vector_db import setup_vector_db

# Constants
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclasses.dataclass
class PerformanceData:
    """Represents performance data."""
    steps_times: Dict[str, float]

# Global dictionary to store step times
performance_data: Dict[str, float] = {}

class ProcessingError(Exception):
    """Custom exception for processing errors."""

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
def step_1_load_and_split_pdf(pdf_file: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Loads a PDF and splits it into chunks."""
    logging.info("Loading and splitting PDF: %s", pdf_file)

    if not isinstance(pdf_file, str):
        raise ValueError("pdf_file must be a string")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer")

    documents = load_pdf(pdf_file)

    if not documents:
        raise ProcessingError("Error loading PDF.")

    chunks = split_text (documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ProcessingError("Error splitting text.")
    logging.info("Text split into %s chunks", len(chunks))
    
    return chunks

@timed_function
def step_2_setup_vector_database(chunks: List, embedding_model: str, collection_name: str):
    """Sets up the vector database."""
    logging.info("Setting up vector database...")
    if not isinstance(chunks, list):
        raise ValueError("chunks must be a list")
    if not isinstance(embedding_model, str):
        raise ValueError("embedding_model must be a string")
    if not isinstance(collection_name, str):
        raise ValueError("collection_name must be a string")
    
    vector_db = setup_vector_db (chunks, embedding_model)
    if not vector_db:
        raise ProcessingError("Error setting up vector database.")
    return vector_db

@timed_function
def step_3_load_language_model(model_name: str):
    """Loads the language model."""
    logging.info("Loading language model: %s", model_name)
    if not isinstance(model_name, str):
        raise ValueError("model_name must be a string")
    llm = load_llm(model_name)
    if not llm:
        raise ProcessingError(f"Error loading language model: {model_name}")
    return llm

@timed_function
def step_4_setup_retrieval_system(vector_db, llm):
    """Sets up the retrieval system."""
    logging.info("Setting up retrieval system...")
    if vector_db is None:
        raise ValueError("vector_db cannot be None")
    if llm is None:
        raise ValueError("llm cannot be None")
    query_prompt = get_query_prompt()
    retriever = setup_retriever(vector_db, llm, query_prompt)
    if not retriever:
        raise ProcessingError("Error setting up retrieval system.")
    return retriever

def execute_llm_query(retriever, llm, question: str) -> Dict:
    """Executes a query and returns the result and elapsed time."""
    logging.info('Executing query: %s', question)
    if retriever is None:
        raise ValueError("retriever cannot be None")
    if llm is None:
        raise ValueError("llm cannot be None")
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    template = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the following context: {context}\nQuestion: {question}")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )
    result = chain.invoke(input=(question,))
    return {"question": question, "answer": result}

def load_config():
    """Loads configuration from environment variables."""
    load_dotenv(override=True)
    pdf_file = os.getenv("PDF_FILE")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    collection_name = os.getenv("COLLECTION_NAME")
    model_name = os.getenv("MODEL_NAME")

    if not all([pdf_file, embedding_model, collection_name, model_name]):
        raise ValueError("Missing environment variables")

    logging.info("PDF_FILE: %s", pdf_file)
    logging.info("EMBEDDING_MODEL: %s", embedding_model)
    logging.info("COLLECTION_NAME: %s", collection_name)
    logging.info("MODEL_NAME: %s", model_name)

    return pdf_file, embedding_model, collection_name, model_name

@timed_function
def step_5_process_queries(retriever, llm) -> List[Dict]:
    """Processes the queries and returns the results."""
    query_results = [
        execute_llm_query(retriever, llm, "Genera un resumen del documento"),
        # execute_query(retriever, llm, "Dime el titulo del documento")
    ]
    questions_and_answers: List[Dict] = []
    for result in query_results:
        questions_and_answers.append({
            "question": result["question"],
            "answer": result["answer"]
        })
    return questions_and_answers

def create_data_payload(
    system_info: SystemInfo,
    model_name: str,
    embedding_model: str,
    questions_and_answers: List[Dict]
    ) -> DataPayload:
    """Creates the DataPayload object."""
    model_info_obj = ModelInfo(
        model_name=model_name,
        embedding_model=embedding_model
    )

    performance_data_obj = PerformanceData(steps_times=performance_data)

    data_payload = DataPayload(
        server_name=socket.gethostname(),
        timestamp="",
        server_data=system_info,
        performance_data=performance_data_obj,
        model_info=model_info_obj,
        questions_and_answers=questions_and_answers
    )
    return data_payload

def main():
    """Main function to execute the processing pipeline."""
    try:
        system_info: SystemInfo = obtener_info_equipo()
        pdf_file, embedding_model, collection_name, model_name = load_config()

        chunks = step_1_load_and_split_pdf(pdf_file, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        vector_db = step_2_setup_vector_database(chunks, embedding_model, collection_name)
        llm = step_3_load_language_model(model_name)
        retriever = step_4_setup_retrieval_system(vector_db, llm)
        questions_and_answers = step_5_process_queries(retriever, llm)
        data_payload = create_data_payload(
            system_info,
            model_name,
            embedding_model,
            questions_and_answers
        )
        save_data(data_payload)

    except ProcessingError as e:
        logging.error("A processing error occurred: %s", e)
        sys.exit(1)
    except ValueError as e:
        logging.error("A value error occurred: %s", e)
        sys.exit(1)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
