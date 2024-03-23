# RAGatouille API Server

RAGatouille API is a FastAPI application designed to provide an efficient and scalable way to perform operations such as document indexing, searching, reranking, and encoding with the RAGPretrainedModel. It leverages the power of FastAPI to offer high performance and easy-to-use RESTful endpoints.

## Features

- **Document Indexing and Management**: Easily index and manage documents with comprehensive endpoints for adding, updating, and deleting documents.
- **Search and Rerank Functionality**: Perform advanced search queries and rerank search results to meet specific requirements.
- **Document Encoding and Retrieval**: Encode documents for efficient storage and retrieval, and search through encoded documents.
- **RESTful API**: Clear and concise endpoints adhering to RESTful standards, making the API easy to consume.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv

### Installation

1. Clone the repository:

   ```bash
   https://github.com/bclavie/RAGatouille.git
   cd RAGatouille/serve
   ```
2. Set up a virtual environment (optional but recommended): <br>
   Windows
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```
   Others
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or use Poetry
   ```bash
   poetry install
   ```
4. Copy .env.example to .env:
   ```bash
   cp .env.example .env
   ```

### Running the Application
Start the FastAPI server with:

```bash
uvicorn ragatouille_serve.main:app --reload
```
or start with Poetry
```bash
poetry run uvicorn ragatouille_serve.main:app --reload
```
The --reload flag enables hot reloading during development.

### Accessing the API Documentation
Once the server is running, you can view the auto-generated Swagger UI documentation by navigating to:
```bash
http://127.0.0.1:8000/docs
```

## API Endpoints

### Index
- **POST /api/v1/index/:** Create an index with documents.
### Document Management
- **POST /api/v1/documents/search/:** Search documents.
- _**[ISSUE] POST /api/v1/documents/add/:** Add documents to an existing index._
- _**[ISSUE] DELETE /api/v1/documents/delete/:** Delete documents from an index._
### Encode
- **POST /api/v1/encode/:** Encode documents.
- **POST /api/v1/encode/search/:** Search through encoded documents.
- **POST /api/v1/encode/clear/:** Clear all encoded documents.
### Rerank
- **POST /api/v1/rerank/:** Rerank a set of documents based on a query.