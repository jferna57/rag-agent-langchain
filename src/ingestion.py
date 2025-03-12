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
        logging.info("PDF %s cargado correctamente", file_path)
        return data
    except FileNotFoundError:
        logging.error("El archivo %s no fue encontrado.", file_path)
    except ImportError as e:
        logging.error("Biblioteca faltante: %s", e)
    except Exception as e:
        logging.error("Error cargando el PDF: %s", e)
    return None