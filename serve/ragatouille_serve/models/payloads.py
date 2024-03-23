from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

# Search Query Payload
class SearchQuery(BaseModel):
    query: Union[str, List[str]]
    index_name: Optional[str] = None
    k: int = Field(default=10, ge=1, description="Number of results to return")
    force_fast: bool = False
    zero_index_ranks: bool = False

# Index Management Payloads
class IndexQuery(BaseModel):
    collection: List[str]
    document_ids: Optional[List[str]] = None
    document_metadatas: Optional[List[Dict[str, Any]]] = None
    index_name: Optional[str] = None
    overwrite_index: Union[bool, str] = True
    max_document_length: int = 256
    split_documents: bool = True

class AddToIndexQuery(BaseModel):
    new_collection: List[str]
    new_document_ids: Optional[List[str]] = None
    new_document_metadatas: Optional[List[Dict[str, Any]]] = None
    index_name: Optional[str] = None
    split_documents: bool = True

class DeleteFromIndexQuery(BaseModel):
    document_ids: List[str]
    index_name: Optional[str] = None

# Reranking Payload
class RerankQuery(BaseModel):
    query: Union[str, List[str]]
    documents: List[str]
    k: int = Field(default=10, ge=1, description="Number of results to return")
    zero_index_ranks: bool = False
    bsize: int = Field(default=64, ge=1, description="Batch size for reranking")

# Encoding Payload
class EncodeQuery(BaseModel):
    documents: List[str]
    bsize: int = Field(default=32, ge=1, description="Batch size for encoding")
    document_metadatas: Optional[List[Dict[str, Any]]] = None
    verbose: bool = True
    max_document_length: Union[str, int] = "auto"

# Search Encoded Documents Payload
class SearchEncodedDocsQuery(BaseModel):
    query: Union[str, List[str]]
    k: int = Field(default=10, ge=1, description="Number of results to return")
    bsize: int = Field(default=32, ge=1, description="Batch size for searching")

# Clear Encoded Documents Payload
class ClearEncodedDocsQuery(BaseModel):
    force: bool = True
