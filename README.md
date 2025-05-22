# GraphRAG

GraphRAG is a Python-based Retrieval-Augmented Generation (RAG) system that uses a knowledge graph to enhance document retrieval and query answering.

## File Summaries

### `graph_rag.py`
Main entry point for the GraphRAG system. It initializes the system, processes documents, and handles user queries.

### `graph_rag_streamlit.py`
Provides a Streamlit-based user interface for interacting with the GraphRAG system. Users can input web links and queries to fetch and process information.

### `context_navigator.py`
Handles querying the knowledge graph and retrieving relevant answers. It includes methods for context expansion and answer validation.

### `semantic_mapper.py`
Defines the KnowledgeGraph class, which builds and manages the knowledge graph. It includes methods for adding nodes, creating embeddings, and extracting concepts.

### `data_splitter.py`
Processes documents by splitting them into chunks and creating embeddings. It also manages a vector store for efficient document retrieval.

### `idea_extractor.py`
Defines a Pydantic model for representing a list of concepts extracted from documents.

### `response_validator.py`
Defines a Pydantic model for validating whether a context provides a complete answer to a query.

### `pyproject.toml`
Specifies project metadata, dependencies, and Python version requirements.

### `requirements.txt`
Lists all the Python packages required to run the project.

## Installation Guide

### Prerequisites
- Python 3.12 

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd GraphRAG
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Startup Guide

### Using the Startup Script
Run the following command to install dependencies and start the Uvicorn server:
```bash
python startup.py
```

### Manual Startup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main file directly:
   ```bash
   python graph_rag.py
   ```

### Streamlit Execution
Run the following command to start the Streamlit-based user interface:
```bash
streamlit run graph_rag_streamlit.py
```
This will launch a web-based interface where you can input web links and queries to interact with the GraphRAG system.

## Main File
The main file for the project is `graph_rag.py`. It initializes the GraphRAG system and provides methods for document processing and querying.
