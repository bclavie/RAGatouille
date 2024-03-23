from fastapi import FastAPI
from ragatouille_serve.api.api_v1 import api_router
from ragatouille_serve.core.config import Settings
from ragatouille_serve.utils.logging_config import setup_logging
import logging

# Initialize FastAPI app and settings
settings = Settings()
app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")

# FastAPI shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
