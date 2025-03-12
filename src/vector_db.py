import os
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def vector_db_exists(vector_db_path) -> bool:
    """
    Verifica si la base de datos vectorial ya existe en el directorio especificado.

    Retorna:
        bool: True si la base de datos existe, False en caso contrario.
    """
    return os.path.exists(vector_db_path)

def setup_vector_db(vector_db_path, chunks, embedding_model, collection_name):
    """
    Configura una base de datos vectorial utilizando Chroma y OllamaEmbeddings.

    Si la base de datos ya existe, se carga; de lo contrario, se crea una nueva.

    Parámetros:
        chunks (list): Lista de documentos fragmentados.
        embedding_model (str): Nombre del modelo de Ollama para generar embeddings.
        collection_name (str): Nombre de la colección en la base de datos vectorial.

    Retorna:
        Chroma: Instancia de la base de datos vectorial configurada.
        None: Si ocurre un error durante la configuración.
    """
    # Verifica si la base de datos ya existe
    if vector_db_exists(vector_db_path):
        print("La base de datos ya existe. Cargando la base de datos existente...")
        try:
            # Carga la base de datos existente
            vector_db = Chroma(
                persist_directory=vector_db_path, 
                embedding_function=OllamaEmbeddings(model=embedding_model),
                collection_name=collection_name
            )
            return vector_db
        except Exception as e:
            print(f"Error al cargar la base de datos existente: {e}")
            return None
    else:
        try:
            # Descarga el modelo de embeddings de Ollama
            ollama.pull(embedding_model)
            # Crea una nueva base de datos y añade los documentos
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model=embedding_model),
                collection_name=collection_name,
                persist_directory=vector_db_path  # Especifica el directorio de persistencia
            )
            print("Base de datos vectorial configurada correctamente")
            return vector_db
        except Exception as e:
            print(f"Error configurando la base de datos vectorial: {e}")
            return None
