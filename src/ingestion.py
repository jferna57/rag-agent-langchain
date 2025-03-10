from langchain_community.document_loaders import UnstructuredPDFLoader

def load_pdf(file_path: str):
    """
    Carga un archivo PDF y extrae su contenido utilizando UnstructuredPDFLoader.

    Parámetros:
    - file_path (str): La ruta del archivo PDF.

    Retorna:
    - data (list) o None en caso de error.
    """
    try:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        print("PDF cargado correctamente")
        return data
    except FileNotFoundError:
        print("El archivo no fue encontrado.")
    except ImportError as e:
        print(f"Biblioteca faltante: {e}")
    except Exception as e:
        print(f"Error cargando el PDF: {e}")
    return None