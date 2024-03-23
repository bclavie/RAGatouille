from fastapi import APIRouter, Depends, HTTPException, status
from ragatouille_serve.models.payloads import RerankQuery
from ragatouille_serve.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def rerank(query: RerankQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting rerank operation for query: '{query.query}' with {len(query.documents)} documents.")
        results = rag.rerank(
            query=query.query,
            documents=query.documents,
            k=query.k,
            zero_index_ranks=query.zero_index_ranks,
            bsize=query.bsize
        )
        logger.info(f"Rerank operation completed successfully for query: '{query.query}'.")
        return results
    except Exception as e:
        logger.error(f"Rerank operation failed for query: '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rerank operation failed"
        )
