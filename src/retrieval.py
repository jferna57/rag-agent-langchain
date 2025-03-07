from langchain.retrievers.multi_query import MultiQueryRetriever

def setup_retriever(vector_db, llm, query_prompt):
    """
    Configura un sistema de recuperación de múltiples consultas utilizando un modelo de lenguaje (LLM) y una base de datos vectorial.

    Esta función configura el sistema de recuperación usando la clase `MultiQueryRetriever` de la biblioteca `langchain.retrievers.multi_query`.
    Se utiliza para realizar consultas a una base de datos vectorial a través de un modelo de lenguaje, generando múltiples versiones alternativas
    de la consulta original para obtener resultados más relevantes.

    Parámetros:
    - vector_db (Retriever): Instancia de un objeto recuperador de base de datos vectorial.
    - llm (ChatOllama): Modelo de lenguaje (LLM) que se utilizará para generar consultas alternativas.
    - query_prompt (PromptTemplate): Plantilla de consulta que especifica cómo generar versiones alternativas de la pregunta original.

    Retorna:
    - retriever (MultiQueryRetriever): Una instancia del sistema de recuperación configurado correctamente.
    - None: Si ocurre un error durante la configuración del sistema de recuperación.

    Excepciones:
    - Si ocurre un error durante la configuración del sistema de recuperación, se captura y muestra el mensaje de error.

    Ejemplo de uso:
    retriever = setup_retriever(vector_db, llm, query_prompt)
    """
    try:
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=query_prompt
        )
        print("Sistema de recuperación configurado correctamente")
        return retriever
    except Exception as e:
        print(f"Error configurando el sistema de recuperación: {e}")
        return None