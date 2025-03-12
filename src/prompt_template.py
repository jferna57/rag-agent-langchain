from langchain.prompts import PromptTemplate

def get_query_prompt():
    """
    Crea una plantilla de consulta para generar versiones alternativas de una pregunta del usuario.

    Esta función utiliza la clase 'PromptTemplate' de la biblioteca 'langchain.prompts' para definir una plantilla
    que toma una pregunta del usuario como entrada y genera cinco versiones alternativas de la misma pregunta.
    Estas versiones se utilizarán para realizar consultas en una base de datos vectorial y recuperar documentos
    relevantes basados en la pregunta original.

    Parámetros:
    - Ninguno.

    Retorna:
    - PromptTemplate: Una instancia de 'PromptTemplate' que puede ser utilizada para generar preguntas alternativas
      basadas en la pregunta original.

    Ejemplo de uso:
    query_prompt = get_query_prompt()
    """
    return PromptTemplate(
        input_variables=["question"],
        template="""Genera cinco versiones alternativas de la pregunta del usuario
        para recuperar documentos relevantes desde la base de datos vectorial.
        Pregunta original: {question}""",
    )
