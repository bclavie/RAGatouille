from fastapi import APIRouter, Depends, HTTPException, status
from app.models.payloads import AddToIndexQuery, DeleteFromIndexQuery
from app.core.rag_model import get_rag_model
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
        return {"message": "Documents deleted from index successfully"}
    except Exception as e:
        logger.error(f"Failed to delete documents from index: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete documents from index")
