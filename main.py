import time
from dotenv import load_dotenv
import os

from src.config import *
from src.ingestion import load_pdf
from src.chunking import split_text
from src.performance import log_times
from src.vector_db import setup_vector_db
from src.model_loader import load_llm
from src.retrieval import setup_retriever
from src.prompt_template import get_query_prompt

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

if __name__ == '__main__':
    steps_times = {}
    
    # Carga del PDF
    start_time = time.time()
    documents = load_pdf(PDF_PATH)
    if not documents:
        exit(1)
    end_time = time.time()
    steps_times["1. Cargar PDF"] = end_time - start_time
    
    # División en fragmentos
    start_time = time.time()
    chunks = split_text(documents)
    if not chunks:
        exit(1)
    end_time = time.time()
    steps_times["2. Dividir texto"] = end_time - start_time
    
    # Configuración de la base de datos vectorial
    start_time = time.time()
    vector_db = setup_vector_db(chunks, EMBEDDING_MODEL, COLLECTION_NAME)
    if not vector_db:
        exit(1)
    end_time = time.time()
    steps_times["3. Configurar base de datos vectorial"] = end_time - start_time

    # Carga del modelo LLM
    start_time = time.time()
    llm = load_llm()
    if not llm:
        exit(1)
    end_time = time.time()
    steps_times["4. Cargar modelo LLM"] = end_time - start_time

    # Configuración del sistema de recuperación
    start_time = time.time()
    query_prompt = get_query_prompt()
    retriever = setup_retriever(vector_db, llm, query_prompt)
    if not retriever:
        exit(1)
    end_time = time.time()
    steps_times["5. Configurar sistema de recuperación"] = end_time - start_time

    # Ejecución de una consulta de ejemplo
    start_time = time.time()
    template = ChatPromptTemplate.from_template("Answer the question based ONLY on the following context: {context}\nQuestion: {question}")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(input=("¿De qué trata el documento?",))
    end_time = time.time()
    steps_times["6. Ejecutar consulta - Resumen"] = end_time - start_time
    print("Respuesta: ", result)

    # Ejecución de una consulta de ejemplo 2
    start_time = time.time()
    result = chain.invoke(input=("¿Quien es el autor del documento?",))
    steps_times["7. Ejecutar consulta - autor"] = end_time - start_time
    print("Respuesta: ", result)
    
    # Mostrar resumen de tiempos
    log_times(steps_times)