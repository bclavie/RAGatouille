from fastapi import APIRouter, Depends, HTTPException, status
from ragatouille_serve.models.payloads import EncodeQuery, SearchEncodedDocsQuery, ClearEncodedDocsQuery
from ragatouille_serve.core.rag_model import get_rag_model
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/")
async def encode(query: EncodeQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting document encoding. Batch size: {query.bsize}, Verbose: {query.verbose}")
        rag.encode(
            documents=query.documents,
            bsize=query.bsize,
            document_metadatas=query.document_metadatas,
            verbose=query.verbose,
            max_document_length=query.max_document_length
        )
        logger.info(f"Documents encoded successfully. Total documents: {len(query.documents)}")
        return {"message": "Documents encoded successfully"}
    except Exception as e:
        logger.error(f"Failed to encode documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to encode documents"
        )

@router.post("/clear")
async def clear_encoded_docs(query: ClearEncodedDocsQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Clearing encoded documents. Force: {query.force}")
        rag.clear_encoded_docs(force=query.force)
        logger.info("Encoded documents cleared successfully.")
        return {"message": "Encoded documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear encoded documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to clear encoded documents"
        )

@router.post("/search")
async def search_encoded_docs(query: SearchEncodedDocsQuery, rag=Depends(get_rag_model)):
    try:
        logger.info(f"Starting search in encoded documents. Query: '{query.query}'")
        results = rag.search_encoded_docs(
            query=query.query,
            k=query.k,
            bsize=query.bsize
        )
        logger.info(f"Search in encoded documents completed successfully. Query: '{query.query}'")
        return results
    except Exception as e:
        logger.error(f"Search in encoded documents failed for query: '{query.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search in encoded documents failed"
        )
