import logging

from langchain_ollama import ChatOllama

def load_llm(model_name:str):
    """
    Carga un modelo de lenguaje (LLM) utilizando Ollama.

    Esta función crea una instancia del modelo de lenguaje utilizando la clase 'ChatOllama' de la biblioteca 'langchain_ollama'.
    El modelo se carga con el nombre de modelo proporcionado en la configuración (variable `MODEL_NAME`).

    Parámetros:
    - Ninguno.

    Retorna:
    - llm (ChatOllama): Una instancia del modelo de lenguaje cargado correctamente.
    - None: Si ocurre un error durante la carga del modelo.

    Excepciones:
    - En caso de un error durante la carga del modelo de lenguaje, la función captura y muestra el mensaje de error.

    Ejemplo de uso:
    llm = load_llm()
    """
    try:
        llm = ChatOllama(model=model_name)
        logging.info("Modelo LLM %s cargado correctamente", model_name)
        return llm
    except Exception as e:
        logging.error("Error cargando el modelo LLM %s : %s", model_name, e)
        return None
