import logging

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
        logging.info(f"PDF {file_path} cargado correctamente")
        return data
    except FileNotFoundError:
        logging.error(f"El archivo {file_path} no fue encontrado.")
    except ImportError as e:
        logging.error(f"Biblioteca faltante: {e}")
    except Exception as e:
        logging.error(f"Error cargando el PDF: {e}")
    return None