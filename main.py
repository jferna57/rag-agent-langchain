import os
from dotenv import load_dotenv
import socket

from src.data import save_data, DataPayload, SystemInfo, ModelInfo, PerformanceData, QuestionAnswerPair
from src.ingestion import load_pdf
from src.chunking import split_text
from src.performance import log_times, timed_execution
from src.vector_db import setup_vector_db
from src.model_loader import load_llm
from src.retrieval import setup_retriever
from src.prompt_template import get_query_prompt

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.utils import obtener_info_equipo

from typing import List

def main():
    """Función principal para ejecutar el flujo de procesamiento."""
    system_info : SystemInfo = obtener_info_equipo()

    # Cargar el archivo .env
    load_dotenv(override=True)

    PDF_FILE = os.getenv("PDF_FILE")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    MODEL_NAME = os.getenv("MODEL_NAME")
    
    # Imprime las variables de entorno
    print(f"PDF_FILE: {PDF_FILE}")
    print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH}")
    print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    
    steps_times = {}
    questions_and_answers = []  # Lista para almacenar las preguntas y respuestas


    # 1. Cargar PDF
    documents, elapsed = timed_execution(load_pdf, PDF_FILE)
    if not documents:
        print("Error al cargar el PDF.")
        return
    steps_times["Cargar PDF"] = elapsed

    # 2. Dividir texto
    chunks, elapsed = timed_execution(split_text, documents)
    if not chunks:
        print("Error al dividir el texto.")
        return
    steps_times["Dividir texto"] = elapsed

    # 3. Configurar base de datos vectorial
    vector_db, elapsed = timed_execution(setup_vector_db, VECTOR_DB_PATH, chunks, EMBEDDING_MODEL, COLLECTION_NAME)
    if not vector_db:
        print("Error al configurar la base de datos vectorial.")
        return
    steps_times["Configurar base de datos vectorial"] = elapsed

    # 4. Cargar modelo LLM
    
    llm, elapsed = timed_execution(load_llm, MODEL_NAME)
    if not llm:
        print(f"Error al cargar el modelo LLM: {MODEL_NAME}")
        return
    steps_times[f"Cargar modelo LLM"] = elapsed

    # 5. Configurar sistema de recuperación
    query_prompt = get_query_prompt()
    retriever, elapsed = timed_execution(setup_retriever, vector_db, llm, query_prompt)
    if not retriever:
        print("Error al configurar el sistema de recuperación.")
        return
    steps_times["Configurar sistema retriever"] = elapsed

    # 6. Ejecutar consulta - Resumen
    template = ChatPromptTemplate.from_template("Answer the question based ONLY on the following context: {context}\nQuestion: {question}")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )
    question_1 = "Genera un resumen del documento"
    result_1, elapsed = timed_execution(chain.invoke, input=(question_1,))
    steps_times["Ejecutar consulta Resumen"] = elapsed
    print("Respuesta: ", result_1)
    
    # Guardar la pregunta y la respuesta
    questions_and_answers.append({"question": question_1, "answer": result_1})

    # 7. Ejecutar consulta - autor
    question_2 = "Dime el titulo del documento"
    result_2, elapsed = timed_execution(chain.invoke, input=(question_2,))
    steps_times["Ejecutar consulta autor"] = elapsed
    print("Respuesta: ", result_2)
        # Guardar la pregunta y la respuesta
    questions_and_answers.append({"question": question_2, "answer": result_2})
    
    # Mostrar resumen de tiempos
    times_json = log_times(steps_times)

    # Create ModelInfo object
    model_info_obj = ModelInfo(
        model_name=MODEL_NAME,
        embedding_model=EMBEDDING_MODEL
    )

    # Create PerformanceData object
    performance_data_obj = PerformanceData(
        steps_times=times_json
    )

    # Create QuestionAnswerPair objects
    question_answer_pairs: List[QuestionAnswerPair] = [QuestionAnswerPair(question=qa["question"], answer=qa["answer"]) for qa in questions_and_answers]
    
    # Create DataPayload object
    data_payload = DataPayload(
        server_name=socket.gethostname(),
        timestamp="",
        server_data=system_info,
        performance_data=performance_data_obj,
        model_info=model_info_obj,
        questions_and_answers=question_answer_pairs
    )
    save_data(data_payload)

if __name__ == '__main__':
    main()