import os

# Obtener las variables de entorno
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "simple-rag")
PDF_PATH = os.getenv("PDF_PATH", "./data/doc3.pdf")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_store")