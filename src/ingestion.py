from langchain_community.document_loaders import UnstructuredPDFLoader

def load_pdf(file_path: str):
    """
    Carga un archivo PDF y extrae su contenido.

    Esta función utiliza el 'UnstructuredPDFLoader' de la biblioteca 'langchain_community.document_loaders'
    para cargar un archivo PDF desde una ruta de archivo proporcionada y extraer su contenido en un formato 
    adecuado para su posterior procesamiento.

    Parámetros:
    - file_path (str): La ruta del archivo PDF que se desea cargar.

    Retorna:
    - data (list): Una lista que contiene el contenido del archivo PDF, extraído y procesado.
    - None: Si ocurre un error durante la carga del archivo PDF.

    Excepciones:
    - En caso de un error durante la carga o el procesamiento del PDF, la función captura y muestra el mensaje de error.

    Ejemplo de uso:
    data = load_pdf("ruta/al/documento.pdf")
    """
    try:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        print("PDF cargado correctamente")
        return data
    except Exception as e:
        print(f"Error cargando el PDF: {e}")
        return None