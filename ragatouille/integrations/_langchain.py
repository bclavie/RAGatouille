from typing import Any, Optional, List, Sequence
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


class RAGatouilleLangChainRetriever(BaseRetriever):
    model: Any
    kwargs: dict = {}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search(query, **self.kwargs)
        return [Document(page_content=doc["content"]) for doc in docs]


class RAGatouilleLangChainCompressor(BaseDocumentCompressor):
    model: Any
    kwargs: dict = {}

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        k: int = 5,
        callbacks: Optional[Callbacks] = None,
        **kwargs,
    ) -> Any:
        """Rerank a list of documents relevant to a query."""
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.model.rerank(
            query=query,
            documents=_docs,
            k=k,
            **self.kwargs,
        )
        final_results = []
        for r in results:
            doc = doc_list[r["result_index"]]
            doc.metadata["relevance_score"] = r["score"]
            final_results.append(doc)
        return final_results
