from fastapi import APIRouter
from ragatouille_serve.api.endpoints import index, document_management, rerank
from ragatouille_serve.api.endpoints import encode

# Versioned API router
api_router = APIRouter()

# Include routers from different endpoint modules
api_router.include_router(index.router, tags=["Index"], prefix="/index")
api_router.include_router(document_management.router, tags=["Document Management"], prefix="/documents")
api_router.include_router(encode.router, tags=["Encode"], prefix="/encode")
api_router.include_router(rerank.router, tags=["Rerank"], prefix="/rerank")
