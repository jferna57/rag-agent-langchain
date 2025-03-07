from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents, chunk_size=1200, chunk_overlap=300):
    """
    Divide un conjunto de documentos en fragmentos de texto más pequeños.

    Esta función utiliza el 'RecursiveCharacterTextSplitter' de la biblioteca 'langchain_text_splitters'
    para dividir un documento grande en fragmentos más pequeños, con un tamaño definido por 'chunk_size'.
    Además, puede superponerse entre fragmentos adyacentes, lo que se controla mediante 'chunk_overlap'.

    Parámetros:
    - documents (list): Lista de documentos (strings o objetos) que se van a dividir.
    - chunk_size (int, opcional): El tamaño máximo de cada fragmento de texto. 
      El valor por defecto es 1200 caracteres.
    - chunk_overlap (int, opcional): La cantidad de superposición entre fragmentos consecutivos. 
      El valor por defecto es 300 caracteres.

    Retorna:
    - list: Una lista de fragmentos de texto divididos.
    - None: Si ocurre algún error durante el proceso de división.

    Excepciones:
    - En caso de un error durante la división, la función captura y muestra el mensaje de error.

    Ejemplo de uso:
    chunks = split_text(documents, chunk_size=1000, chunk_overlap=200)
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Documento dividido en {len(chunks)} fragmentos")
        return chunks
    except Exception as e:
        print(f"Error dividiendo el texto: {e}")
        return None