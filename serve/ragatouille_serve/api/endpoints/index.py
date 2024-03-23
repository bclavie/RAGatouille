from fastapi import APIRouter, Depends, HTTPException, status
from ragatouille_serve.models.payloads import IndexQuery
from ragatouille_serve.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def create_index(index_query: IndexQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting index creation. Index name: {index_query.index_name}, Overwrite: {index_query.overwrite_index}")
        index_path = rag.index(
            collection=index_query.collection,
            document_ids=index_query.document_ids,
            document_metadatas=index_query.document_metadatas,
            index_name=index_query.index_name,
            overwrite_index=index_query.overwrite_index,
            max_document_length=index_query.max_document_length,
            split_documents=index_query.split_documents
        )
        logger.info(f"Index created successfully. Index path: {index_path}")
        return {"index_path": index_path}
    except Exception as e:
        logger.error(f"Failed to create index: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to create index"
        )
