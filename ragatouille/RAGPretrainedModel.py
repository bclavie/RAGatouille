from typing import Callable, Optional, Union, Any
from pathlib import Path
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter
from ragatouille.models import LateInteractionModel, ColBERT
from ragatouille.integrations import (
    RAGatouilleLangChainRetriever,
    RAGatouilleLangChainCompressor,
)


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
        instance.model = ColBERT(pretrained_model_name_or_path, n_gpu, verbose=verbose)
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
        collection: list[str],
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
            index_name (str): The name of the index that will be built.
            overwrite_index (bool): Whether to overwrite an existing index with the same name.

        Returns:
            index (str): The path to the index that was built.
        """
        if split_documents or preprocessing_fn is not None:
            self.corpus_processor = CorpusProcessor(
                document_splitter_fn=document_splitter_fn if split_documents else None,
                preprocessing_fn=preprocessing_fn,
            )
            collection = self.corpus_processor.process_corpus(
                collection,
                chunk_size=max_document_length,
            )
        overwrite = "reuse"
        if overwrite_index:
            overwrite = True
        return self.model.index(
            collection,
            index_name,
            max_document_length=max_document_length,
            overwrite=overwrite,
        )

    def add_to_index(
        self,
        new_documents: list[str],
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

    def rerank(
        self,
        query: str | list[str],
        documents: list[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: int = 64,
    ):
        """Re-rank documents via ColBERT. Functionally equivalent to `direct_search()`.

        Parameters:
            query (Union[str, list[str]]): The query or list of queries to search for.
            k (int): The number of results to return for each query.
            zero_index_ranks (bool): Whether to zero the index ranks of the results. By default, result rank 1 is the highest ranked result
            bsize (int): The batch size to use for re-ranking.

        Returns:
            results (Union[list[dict], list[list[dict]]]): A list of dict containing individual results for each query. If a list of queries is provided, returns a list of lists of dicts. Each result is a dict with keys `content`, `score` and `rank`.

        Individual results are always in the format:
        ```python3
        {"content": "text of the relevant passage", "score": 0.123456, "rank": 1}
        ```
        """

        return self.model.rank(
            query=query,
            documents=documents,
            k=k,
            zero_index_ranks=zero_index_ranks,
            bsize=bsize,
        )

    def as_langchain_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RAGatouilleLangChainRetriever(model=self, kwargs=kwargs)

    def as_langchain_document_compressor(self, **kwargs: Any) -> BaseDocumentCompressor:
        return RAGatouilleLangChainCompressor(model=self, kwargs=kwargs)
