from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, TypeVar, Union
from uuid import uuid4

from huggingface_hub import hf_hub_download, try_to_load_from_cache
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.retrievers import BaseRetriever
from transformers.utils import cached_file

from ragatouille.data.corpus_processor import CorpusProcessor
from ragatouille.data.preprocessors import llama_index_sentence_splitter
from ragatouille.integrations import (
    RAGatouilleLangChainCompressor,
    RAGatouilleLangChainRetriever,
)
from ragatouille.models import ColBERT, LateInteractionModel
from ragatouille.models.utils import upload_index_and_model


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
    index_path: Union[Path, None] = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        index_root: Optional[str] = None,
    ):
        """Load a ColBERT model from a pre-trained checkpoint.

        Parameters:
            pretrained_model_name_or_path (str): Local path or huggingface model name.
            n_gpu (int): Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.
            verbose (int): The level of ColBERT verbosity requested. By default, 1, which will filter out most internal logs.
            index_root (Optional[str]): The root directory where indexes will be stored. If None, will use the default directory, '.ragatouille/'.

        Returns:
            cls (RAGPretrainedModel): The current instance of RAGPretrainedModel, with the model initialised.
        """
        instance = cls()
        instance.model = ColBERT(
            pretrained_model_name_or_path, n_gpu, index_root=index_root, verbose=verbose
        )
        return instance

    @classmethod
    def from_index(
        cls,
        index_path: Union[str, Path],
        n_gpu: int = -1,
        verbose: int = 1,
        local_dir: Optional[str] = None,
    ):
        """Load an Index and the associated ColBERT encoder from an existing document index.

        Parameters:
            index_path (Union[str, Path]): Path to the index (eg., ./my-index) or, The Hugging Face repository ID (e.g., 'username/repo').
            n_gpu (int): Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.
            verbose (int): The level of ColBERT verbosity requested. By default, 1, which will filter out most internal logs.
            local_dir (Optional[str]): Local directory to download the index to, if not present locally.

        Returns:
            cls (RAGPretrainedModel): The current instance of RAGPretrainedModel, with the model and index initialised.
        """
        instance = cls()

        # Check if index_path refers to a path or a huggingface model
        resolved_index_path = cls._resolve_index_path(index_path, local_dir)

        instance.model = ColBERT(
            resolved_index_path, n_gpu, verbose=verbose, load_from_index=True
        )

        return instance

    @staticmethod
    def _resolve_index_path(
        index_path: Union[str, Path], local_dir: Optional[str]
    ) -> Path:
        """
        Resolve whether the index_path is a local path or a Hugging Face model identifier.

        Parameters:
            index_path (Union[str, Path]): The index path or Hugging Face model identifier.
            local_dir (Optional[str]): The local directory to use for storing the index.

        Returns:
            Path: The resolved local path to the index.
        """
        index_path = Path(index_path)
        if local_dir is None:
            local_dir = f".ragatouille/indexes/{str(index_path)}"

        if index_path.is_dir():
            return index_path
        else:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=str(index_path),
                        filename="indexes/",
                        cache_dir=local_dir,
                    )
                )
            except Exception as e:
                raise OSError(
                    f"Index not found locally or in the Hugging Face hub: {index_path}"
                ) from e

    def _process_metadata(
        self,
        document_ids: Optional[Union[TypeVar("T"), List[TypeVar("T")]]],
        document_metadatas: Optional[list[dict[Any, Any]]],
        collection_len: int,
    ) -> tuple[list[str], Optional[dict[Any, Any]]]:
        if document_ids is None:
            document_ids = [str(uuid4()) for i in range(collection_len)]
        else:
            if len(document_ids) != collection_len:
                raise ValueError("document_ids must be the same length as collection")
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("document_ids must be unique")
            if any(not id.strip() for id in document_ids):
                raise ValueError("document_ids must not contain empty strings")
            if not all(isinstance(id, type(document_ids[0])) for id in document_ids):
                raise ValueError("All document_ids must be of the same type")

        if document_metadatas is not None:
            if len(document_metadatas) != collection_len:
                raise ValueError(
                    "document_metadatas must be the same length as collection"
                )
            docid_metadata_map = {
                x: y for x, y in zip(document_ids, document_metadatas)
            }
        else:
            docid_metadata_map = None

        return document_ids, docid_metadata_map

    def index(
        self,
        collection: list[str],
        document_ids: Union[TypeVar("T"), List[TypeVar("T")]] = None,
        document_metadatas: Optional[list[dict]] = None,
        index_name: str = None,
        overwrite_index: Union[bool, str] = True,
        max_document_length: int = 256,
        split_documents: bool = True,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        """Build an index from a list of documents.

        Parameters:
            collection (list[str]): The collection of documents to index.
            document_ids (Optional[list[str]]): An optional list of document ids. Ids will be generated at index time if not supplied.
            index_name (str): The name of the index that will be built.
            overwrite_index (Union[bool, str]): Whether to overwrite an existing index with the same name.
            max_document_length (int): The maximum length of a document. Documents longer than this will be split into chunks.
            split_documents (bool): Whether to split documents into chunks.
            document_splitter_fn (Optional[Callable]): A function to split documents into chunks. If None and by default, will use the llama_index_sentence_splitter.
            preprocessing_fn (Optional[Union[Callable, list[Callable]]]): A function or list of functions to preprocess documents. If None and by default, will not preprocess documents.

        Returns:
            index (str): The path to the index that was built.
        """

        document_ids, docid_metadata_map = self._process_metadata(
            document_ids=document_ids,
            document_metadatas=document_metadatas,
            collection_len=len(collection),
        )

        if split_documents or preprocessing_fn is not None:
            self.corpus_processor = CorpusProcessor(
                document_splitter_fn=document_splitter_fn if split_documents else None,
                preprocessing_fn=preprocessing_fn,
            )
            collection_with_ids = self.corpus_processor.process_corpus(
                collection,
                document_ids,
                chunk_size=max_document_length,
            )
        else:
            collection_with_ids = [
                {"document_id": x, "content": y}
                for x, y in zip(document_ids, collection)
            ]

        pid_docid_map = {
            index: item["document_id"] for index, item in enumerate(collection_with_ids)
        }
        collection = [x["content"] for x in collection_with_ids]
        return self.model.index(
            collection,
            pid_docid_map=pid_docid_map,
            docid_metadata_map=docid_metadata_map,
            index_name=index_name,
            max_document_length=max_document_length,
            overwrite=overwrite_index,
        )

    def add_to_index(
        self,
        new_collection: list[str],
        new_document_ids: Optional[Union[TypeVar("T"), List[TypeVar("T")]]] = None,
        new_document_metadatas: Optional[list[dict]] = None,
        index_name: Optional[str] = None,
        split_documents: bool = True,
        document_splitter_fn: Optional[Callable] = llama_index_sentence_splitter,
        preprocessing_fn: Optional[Union[Callable, list[Callable]]] = None,
    ):
        """Add documents to an existing index.

        Parameters:
            new_collection (list[str]): The documents to add to the index.
            new_document_metadatas (Optional[list[dict]]): An optional list of metadata dicts
            index_name (Optional[str]): The name of the index to add documents to. If None and by default, will add documents to the already initialised one.
        """
        new_document_ids, new_docid_metadata_map = self._process_metadata(
            document_ids=new_document_ids,
            document_metadatas=new_document_metadatas,
            collection_len=len(new_collection),
        )

        if split_documents or preprocessing_fn is not None:
            self.corpus_processor = CorpusProcessor(
                document_splitter_fn=document_splitter_fn if split_documents else None,
                preprocessing_fn=preprocessing_fn,
            )
            new_collection_with_ids = self.corpus_processor.process_corpus(
                new_collection,
                new_document_ids,
                chunk_size=self.model.config.doc_maxlen,
            )
        else:
            new_collection_with_ids = [
                {"document_id": x, "content": y}
                for x, y in zip(new_document_ids, new_collection)
            ]

        new_collection = [x["content"] for x in new_collection_with_ids]

        new_pid_docid_map = {
            index: item["document_id"]
            for index, item in enumerate(new_collection_with_ids)
        }

        self.model.add_to_index(
            new_collection,
            new_pid_docid_map,
            new_docid_metadata_map=new_docid_metadata_map,
            index_name=index_name,
        )

    def delete_from_index(
        self,
        document_ids: Union[TypeVar("T"), List[TypeVar("T")]],
        index_name: Optional[str] = None,
    ):
        """Delete documents from an index by their IDs.

        Parameters:
            document_ids (Union[TypeVar("T"), List[TypeVar("T")]]): The IDs of the documents to delete.
            index_name (Optional[str]): The name of the index to delete documents from. If None and by default, will delete documents from the already initialised one.
        """
        self.model.delete_from_index(
            document_ids,
            index_name=index_name,
        )

    def search(
        self,
        query: Union[str, list[str]],
        index_name: Optional["str"] = None,
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        doc_ids: Optional[list[str]] = None,
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
            results (Union[list[dict], list[list[dict]]]): A list of dict containing individual results for each query. If a list of queries is provided, returns a list of lists of dicts. Each result is a dict with keys `content`, `score`, `rank`, and 'document_id'. If metadata was indexed for the document, it will be returned under the "document_metadata" key.

        Individual results are always in the format:
        ```python3
        {"content": "text of the relevant passage", "score": 0.123456, "rank": 1, "document_id": "x"}
        ```
        or
        ```python3
        {"content": "text of the relevant passage", "score": 0.123456, "rank": 1, "document_id": "x", "document_metadata": {"metadata_key": "metadata_value", ...}}
        ```

        """
        return self.model.search(
            query=query,
            index_name=index_name,
            k=k,
            force_fast=force_fast,
            zero_index_ranks=zero_index_ranks,
            doc_ids=doc_ids,
            **kwargs,
        )

    def rerank(
        self,
        query: Union[str, list[str]],
        documents: list[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: int = 64,
    ):
        """Encode documents and rerank them in-memory. Performance degrades rapidly with more documents.

        Parameters:
            query (Union[str, list[str]]): The query or list of queries to search for.
            documents (list[str]): The documents to rerank.
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

    def encode(
        self,
        documents: list[str],
        bsize: int = 32,
        document_metadatas: Optional[list[dict]] = None,
        verbose: bool = True,
        max_document_length: Union[Literal["auto"], int] = "auto",
    ):
        """Encode documents in memory to be searched through with no Index. Performance degrades rapidly with more documents.

        Parameters:
            documents (list[str]): The documents to encode.
            bsize (int): The batch size to use for encoding.
            document_metadatas (Optional[list[dict]]): An optional list of metadata dicts. Each entry must correspond to a document.
        """
        if verbose:
            print(f"Encoding {len(documents)} documents...")
        self.model.encode(
            documents=documents,
            bsize=bsize,
            document_metadatas=document_metadatas,
            verbose=verbose,
            max_tokens=max_document_length,
        )
        if verbose:
            print("Documents encoded!")

    def search_encoded_docs(
        self,
        query: Union[str, list[str]],
        k: int = 10,
        bsize: int = 32,
    ) -> list[dict[str, Any]]:
        """Search through documents encoded in-memory.

        Parameters:
            query (Union[str, list[str]]): The query or list of queries to search for.
            k (int): The number of results to return for each query.
            batch_size (int): The batch size to use for searching.

        Returns:
            results (list[dict[str, Any]]): A list of dict containing individual results for each query. If a list of queries is provided, returns a list of lists of dicts.
        """
        return self.model.search_encoded_docs(
            queries=query,
            k=k,
            bsize=bsize,
        )

    def clear_encoded_docs(self, force: bool = False):
        """Clear documents encoded in-memory.

        Parameters:
            force (bool): Whether to force the clearing of encoded documents without enforcing a 10s wait time.
        """
        self.model.clear_encoded_docs(force=force)

    def as_langchain_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RAGatouilleLangChainRetriever(model=self, kwargs=kwargs)

    def as_langchain_document_compressor(
        self, k: int = 5, **kwargs: Any
    ) -> BaseDocumentCompressor:
        return RAGatouilleLangChainCompressor(model=self, k=k, kwargs=kwargs)

    def upload_to_huggingface_hub(self, huggingface_repo_name: str) -> None:
        """Upload the given colbert model and"""
        if not self.model:
            raise ValueError(
                "Model is undefined. Specify the model before attempting to upload the index."
            )

        colbert_path = Path(try_to_load_from_cache(repo_id=self.model_name, filename="config.json")).parent
        print(colbert_path)

        upload_index_and_model(
            colbert_path=colbert_path,
            huggingface_repo_name=huggingface_repo_name,
            index_path=self.index_path,
        )
