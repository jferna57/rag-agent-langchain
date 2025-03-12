# Procesamiento de Documentos PDF con Langchain y Ollama

Este proyecto permite cargar documentos PDF, dividirlos en fragmentos, almacenarlos en una base de datos vectorial, cargar un modelo de lenguaje para realizar consultas sobre el contenido del documento y registrar los tiempos de cada paso del procesamiento.

## Tabla de Contenidos

1. [Descripción](#descripción)
2. [Tecnologías Utilizadas](#tecnologías-utilizadas)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Contribución](#contribución)
6. [Licencia](#licencia)

## Descripción

Este proyecto está diseñado para cargar, procesar y consultar documentos PDF de manera eficiente. A través de un conjunto de pasos que incluyen la carga del archivo PDF, la fragmentación del contenido en piezas más pequeñas, el almacenamiento en una base de datos vectorial, la carga de un modelo de lenguaje (LLM) y la configuración de un sistema de recuperación, este sistema facilita la ejecución de consultas sobre el documento procesado.

Los tiempos de ejecución para cada uno de estos pasos se registran y se presentan al final como un resumen de rendimiento.

## Tecnologías Utilizadas

- **Lenguaje de Programación**: Python
- **Frameworks y Librerías**:

  - Ollama: cliente de modelos de IA generativa.
  - LangChain: para manejar los prompts y la cadena de procesamiento.
  - Chroma: para la base de datos vectorial.

- **Modelos de Lenguaje**:
  - Se utiliza un modelo LLM para la ejecución de consultas sobre el documento.

## Instalación

Para instalar y ejecutar este proyecto en tu máquina local, sigue estos pasos:

- Instalar los modelos de ollama siguientes:

```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
```

- Descargar el código fuente

```bash
  git clone https://github.com/jferna57/rag-agent-langchain
  cd rag-agent-langchain
```

- Crear entorno python y instalar las dependencias

```bash
  python -m venv .venv
  source .venv/bin/activate
```

- Instalar las dependencias de la aplicación:

```bash
  pip install --user pipenv
  pipenv install
  pipenv shell
  python main.py
```
