from fastapi import APIRouter, Depends, HTTPException, status
from ragatouille_serve.models.payloads import AddToIndexQuery, DeleteFromIndexQuery, SearchQuery
from ragatouille_serve.core.rag_model import get_rag_model, delete_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_to_index(query: AddToIndexQuery, rag=Depends(get_rag_model)):
    try:
        rag.add_to_index(
            new_collection=query.new_collection,
            new_document_ids=query.new_document_ids,
            new_document_metadatas=query.new_document_metadatas,
            index_name=query.index_name,
            split_documents=query.split_documents
        )
        return {"message": "Documents added to index successfully"}
    except Exception as e:
        logger.error(f"Failed to add documents to index: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add documents to index")

@router.delete("/delete", status_code=status.HTTP_200_OK)
async def delete_from_index(query: DeleteFromIndexQuery, rag=Depends(get_rag_model)):
    try:
        rag.delete_from_index(
            document_ids=query.document_ids,
            index_name=query.index_name
        )
        # delete_rag_model()
        return {"message": "Documents deleted from index successfully"}
    except Exception as e:
        logger.error(f"Failed to delete documents from index: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete documents from index")

@router.post("/search")
async def search(query: SearchQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Initiating search with query: '{query.query}'. Index: '{query.index_name}', k: {query.k}")
        results = rag.search(
            query=query.query, 
            index_name=query.index_name, 
            k=query.k, 
            force_fast=query.force_fast, 
            zero_index_ranks=query.zero_index_ranks
        )
        logger.info(f"Search completed successfully for query: '{query.query}'.")
        return results
    except Exception as e:
        logger.error(f"Search operation failed for query: '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Search operation failed"
        )