from fastapi import Depends
from ragatouille_serve.core.config import settings
from ragatouille import RAGPretrainedModel
import logging

# Set up a logger
logger = logging.getLogger(__name__)

# Singleton pattern to avoid loading the model multiple times
class RAGModel:
    model_instance: RAGPretrainedModel = None

    @classmethod
    def get_instance(cls) -> RAGPretrainedModel:
        if cls.model_instance is None:
            logger.info("Initializing RAGPretrainedModel instance...")
            # Initialize and configure RAGPretrainedModel
            cls.model_instance = RAGPretrainedModel.from_pretrained(pretrained_model_name_or_path=settings.MODEL_NAME, index_root=settings.INDEX_ROOT)
            logger.info("RAGPretrainedModel instance created successfully.")
        else:
            logger.debug("RAGPretrainedModel instance already exists, reusing the existing instance.")
        return cls.model_instance
    
    @classmethod
    def delete_instance(cls):
        if cls.model_instance is not None:
            logger.info("Deleting RAGPretrainedModel instance...")
            cls.model_instance = None
            logger.info("RAGPretrainedModel instance deleted successfully.")
        else:
            logger.debug("RAGPretrainedModel instance does not exist, nothing to delete.")

def get_rag_model() -> RAGPretrainedModel:
    return RAGModel.get_instance()

def delete_rag_model():
    RAGModel.delete_instance()