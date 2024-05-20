from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from voyager import Index, Space, StorageDataType

from .base import HardNegativeMiner


class DenseModels(Enum):
    en_small = "BAAI/bge-small-en-v1.5"
    zh_small = "thenlper/gte-small-zh"
    fr_small = "OrdalieTech/Solon-embeddings-base-0.1"
    other_small = "intfloat/multilingual-e5-small"
    en_base = "BAAI/bge-base-en-v1.5"
    zh_base = "thenlper/gte-base-zh"
    fr_base = "OrdalieTech/Solon-embeddings-base-0.1"
    other_base = "intfloat/multilingual-e5-base"
    en_large = "BAAI/bge-large-en-v1.5"
    zh_large = "thenlper/gte-large-zh"
    fr_large = "OrdalieTech/Solon-embeddings-large-0.1"
    other_large = "intfloat/multilingual-e5-large"


class SimpleMiner(HardNegativeMiner):
    """The simplest approach to hard negatives mining.
    Select the most appropriate, small-sized embedding model for the target language.
    And retrieve random negatives in the top 10-100 results.
    Strong baseline for quick, low-engineering hard negative mining."""

    def __init__(
        self,
        language_code: str,
        model_size: Literal["small", "base", "large"] = "small",
    ) -> None:
        self.n_gpu = torch.cuda.device_count()
        self.target_language = language_code
        self.model_size = model_size
        if language_code not in ["en", "zh", "fr"]:
            language_code = "other"
        self.model_name = f"{language_code}_{model_size}"
        hub_model = DenseModels[self.model_name].value
        print(f"Loading Hard Negative SimpleMiner dense embedding model {hub_model}...")
        self.model = SentenceTransformer(hub_model)
        self.has_index = False
        self.min_rank = 10

    def build_index(
        self,
        collection,
        batch_size: int = 128,
        save_index: bool = False,
        save_path: Union[str, Path] = None,
        force_fp32: bool = True,
    ):
        print(f"Building hard negative index for {len(collection)} documents...")
        if len(collection) > 1000:
            print(100)
            pool = self.model.start_multi_process_pool()
            print(101)
            embeds = self.model.encode_multi_process(
                collection, pool, batch_size=batch_size
            )
            print(102)
            self.model.stop_multi_process_pool(pool)
            print(103)
        else:
            embeds = self.model.encode(collection, batch_size=batch_size)

        print("All documents embedded, now adding to index...")

        self.max_rank = min(110, int(len(collection) // 10))
        self.max_rank = min(self.max_rank, len(collection))

        storage_type = StorageDataType.Float32
        if len(collection) > 500000 and not force_fp32:
            storage_type = StorageDataType.E4M3

        self.voyager_index = Index(
            Space.Cosine,
            num_dimensions=self.model.get_sentence_embedding_dimension(),
            storage_data_type=storage_type,
        )

        self.corpus_map = {i: doc for i, doc in enumerate(collection)}
        id_to_vector = {}
        for i, emb in enumerate(embeds):
            id_to_vector[i] = emb
            self.corpus_map[i] = collection[i]
        del embeds

        self.voyager_index.add_items(
            vectors=[x for x in id_to_vector.values()],
            ids=[x for x in id_to_vector.keys()],
            num_threads=-1,
        )

        del id_to_vector

        if save_index:
            print(f"Saving index to {save_path}...")
            self.export_index(save_path)
        else:
            print("save_index set to False, skipping saving hard negative index")
        print("Hard negative index generated")
        self.has_index = True

    def query_index(self, query, top_k=110):
        results = self.voyager_index.query(
            query, k=min(top_k, self.voyager_index.__len__())
        )
        return results

    def mine_hard_negatives(
        self,
        queries: Union[list[str], str],
        collection: Optional[list[str]] = None,
        save_index: bool = False,
        save_path: Union[str, Path] = None,
        force_fp32: bool = True,
    ):
        if self.has_index is False and collection is not None:
            self.build_index(
                collection,
                save_index=save_index,
                save_path=save_path,
                force_fp32=force_fp32,
            )
        if isinstance(queries, str):
            return self._mine(queries)
        return self._batch_mine(queries)

    def _mine(
        self,
        query: str,
    ):
        q_emb = self.model.encode(query)
        query_results = self.query_index(q_emb, top_k=self.max_rank)
        if len(query_results) > self.min_rank:
            query_results = query_results[self.min_rank : self.max_rank]
        query_results = [self.corpus_map[x] for x in query_results[0]]
        return query_results

    def _batch_mine(
        self,
        queries: list[str],
    ):
        """Separate function to parallelise later on"""
        print(f"Retrieving hard negatives for {len(queries)} queries...")
        results = []
        print("Embedding queries...")
        query_embeddings = self.model.encode(queries, show_progress_bar=True)
        print("Retrieving hard negatives...")
        for q_emb in tqdm(query_embeddings):
            query_results = self.query_index(q_emb, top_k=self.max_rank)
            query_results = query_results[self.min_rank : self.max_rank]
            query_results = [self.corpus_map[x.id] for x in query_results]
            results.append(query_results)
        print(f"""Done generating hard negatives.""")
        return results

    def export_index(self, path: Union[str, Path]) -> bool:
        self.voyager_index.save(path)
        return True
