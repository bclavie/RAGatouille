from typing import Callable, Optional, Union, List, Any, TypeVar
from pathlib import Path
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import (
    CallbackManagerForRetrieverRun,
)
from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter
from ragatouille.models import LateInteractionModel, ColBERT
from uuid import uuid4

class RAGPretrainedModel:
    """
    Wrapper class for a pretrained RAG late-interaction model, and all the associated utilities.
    Allows you to load a pretrained model from disk or from the hub, build or query an index.

    ## Usage

    Load a pre-trained checkpoint:

    ```python
    from ragatouille import RAGPretrainedModel

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    ```

    Load checkpoint from an existing index:

    ```python
    from ragatouille import RAGPretrainedModel

    RAG = RAGPretrainedModel.from_index("path/to/my/index")
    ```

    Both methods will load a fully initialised instance of ColBERT, which you can use to build and query indexes.

    ```python
    RAG.search("How many people live in France?")
    ```
    """

    model_name: Union[str, None] = None
    model: Union[LateInteractionModel, None] = None
    corpus_processor: Optional[CorpusProcessor] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        index_root: Optional[str] = None
    ):
        """Load a ColBERT model from a pre-trained checkpoint.

        Parameters:
            pretrained_model_name_or_path (str): Local path or huggingface model name.
            n_gpu (int): Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.
            verbose (int): The level of ColBERT verbosity requested. By default, 1, which will filter out most internal logs.

        Returns:
            cls (RAGPretrainedModel): The current instance of RAGPretrainedModel, with the model initialised.
        """
        instance = cls()
        instance.model = ColBERT(pretrained_model_name_or_path, n_gpu, index_root=index_root, verbose=verbose)
        return instance

    @classmethod
    def from_index(
        cls, index_path: Union[str, Path], n_gpu: int = -1, verbose: int = 1
    ):
        """Load an Index and the associated ColBERT encoder from an existing document index.

        Parameters:
            index_path (Union[str, path]): Path to the index.
            n_gpu (int): Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.
            verbose (int): The level of ColBERT verbosity requested. By default, 1, which will filter out most internal logs.

        Returns:
            cls (RAGPretrainedModel): The current instance of RAGPretrainedModel, with the model and index initialised.
        """
        instance = cls()
        index_path = Path(index_path)
        instance.model = ColBERT(
            index_path, n_gpu, verbose=verbose, load_from_index=True
        )

        return instance

    def index(
        self,
        documents: list[str],
        document_ids: Union[TypeVar("T"), List[TypeVar("T")]],
        document_metadatas: Optional[list[dict]] = None,
        index_name: str = None,
        overwrite_index: bool = True,
        max_document_length: int = 256,
        split_documents: bool = True,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        """Build an index from a collection of documents.

        Parameters:
            collection (list[str]): The collection of documents to index.
            document_ids (Optional[list[str]]): An optional list of document ids. Ids will be generated at index time if not supplied.
            metadata (Optional[list[dict]]): An optional list of metadata dicts
            index_name (str): The name of the index that will be built.
            overwrite_index (bool): Whether to overwrite an existing index with the same name.

        Returns:
            index (str): The path to the index that was built.
        """

        if len(document_ids) != len(documents):
            raise ValueError("Document IDs and documents must be the same length.")
        
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Document IDs must be unique.")

        if split_documents or preprocessing_fn is not None:
            self.corpus_processor = CorpusProcessor(
                document_splitter_fn=document_splitter_fn if split_documents else None,
                preprocessing_fn=preprocessing_fn,
            )
            collection = self.corpus_processor.process_corpus(
                documents,
                document_ids,
                chunk_size=max_document_length,
            )
        else:
            collection = [{"document_id": x, "content": y} for x, y in zip(document_ids, documents)]
        
        if document_metadatas is not None:
            document_metadata_dict = {x:y for x, y in zip(document_ids, document_metadatas)}
        else:
            document_metadata_dict = None
        
        overwrite = "reuse"
        if overwrite_index:
            overwrite = True
        return self.model.index(
            collection,
            document_metadata_dict,
            index_name,
            max_document_length=max_document_length,
            overwrite=overwrite,
        )

    def add_to_index(
        self,
        new_documents: list[str],
        new_metadata: Optional[list[dict]] = None,
        index_name: Optional[str] = None,
        split_documents: bool = True,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        """Add documents to an existing index.

        Parameters:
            new_documents (list[str]): The documents to add to the index.
            index_name (Optional[str]): The name of the index to add documents to. If None and by default, will add documents to the already initialised one.
        """
        if split_documents or preprocessing_fn is not None:
            self.corpus_processor = CorpusProcessor(
                document_splitter_fn=document_splitter_fn if split_documents else None,
                preprocessing_fn=preprocessing_fn,
            )
            new_documents = self.corpus_processor.process_corpus(
                new_documents,
                chunk_size=self.model.config.doc_maxlen,
            )

        self.model.add_to_index(
            new_documents,
            new_metadata,
            index_name=index_name,
        )

    def search(
        self,
        query: Union[str, list[str]],
        index_name: Optional["str"] = None,
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        **kwargs,
    ):
        """Query an index.

        Parameters:
            query (Union[str, list[str]]): The query or list of queries to search for.
            index_name (Optional[str]): Provide the name of an index to query. If None and by default, will query an already initialised one.
            k (int): The number of results to return for each query.
            force_fast (bool): Whether to force the use of a faster but less accurate search method.
            zero_index_ranks (bool): Whether to zero the index ranks of the results. By default, result rank 1 is the highest ranked result

        Returns:
            results (Union[list[dict], list[list[dict]]]): A list of dict containing individual results for each query. If a list of queries is provided, returns a list of lists of dicts. Each result is a dict with keys `content`, `score` and `rank`.

        Individual results are always in the format:
        ```python3
        {"content": "text of the relevant passage", "score": 0.123456, "rank": 1}
        ```
        """
        return self.model.search(
            query=query,
            index_name=index_name,
            k=k,
            force_fast=force_fast,
            zero_index_ranks=zero_index_ranks,
            **kwargs,
        )

    def as_langchain_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RAGatouilleLangChainRetriever(model=self, kwargs=kwargs)


class RAGatouilleLangChainRetriever(BaseRetriever):

    model: RAGPretrainedModel
    kwargs: dict = {}

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [Document(page_content=doc['content']) for doc in docs]