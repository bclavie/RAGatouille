import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypeVar, Union

import numpy as np
import srsly
import torch
from colbert import Trainer
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint

from ragatouille.models.base import LateInteractionModel
from ragatouille.models.index import ModelIndex, ModelIndexFactory
from ragatouille.evaluation.metrics import resolve_metrics


# TODO: Move all bsize related calcs to `_set_bsize()`


class ColBERT(LateInteractionModel):
    def __init__(
        self,
        pretrained_model_name_or_path: Union[str, Path],
        n_gpu: int = -1,
        index_name: Optional[str] = None,
        verbose: int = 1,
        load_from_index: bool = False,
        training_mode: bool = False,
        index_root: Optional[str] = None,
        **kwargs,
    ):
        self.verbose = verbose
        self.collection = None
        self.pid_docid_map = None
        self.docid_pid_map = None
        self.docid_metadata_map = None
        self.base_model_max_tokens = 510
        if n_gpu == -1:
            n_gpu = 1 if torch.cuda.device_count() == 0 else torch.cuda.device_count()

        self.loaded_from_index = load_from_index

        self.model_index: Optional[ModelIndex] = None
        if load_from_index:
            self.index_path = str(pretrained_model_name_or_path)
            ckpt_config = ColBERTConfig.load_from_index(
                str(pretrained_model_name_or_path)
            )
            self.model_index = ModelIndexFactory.load_from_file(
                self.index_path, index_name, ckpt_config
            )
            self.config = self.model_index.config
            self.run_config = RunConfig(
                nranks=n_gpu, experiment=self.config.experiment, root=self.config.root
            )
            split_root = str(pretrained_model_name_or_path).split("/")[:-1]
            self.config.root = "/".join(split_root)
            self.index_root = self.config.root
            self.checkpoint = self.config.checkpoint
            self.index_name = self.config.index_name
            self._get_collection_files_from_disk(pretrained_model_name_or_path)
            # TODO: Modify root assignment when loading from HF

        else:
            self.index_root = index_root if index_root is not None else ".ragatouille/"
            ckpt_config = ColBERTConfig.load_from_checkpoint(
                str(pretrained_model_name_or_path)
            )
            self.run_config = RunConfig(
                nranks=n_gpu, experiment="colbert", root=self.index_root
            )
            local_config = ColBERTConfig(**kwargs)
            self.config = ColBERTConfig.from_existing(
                ckpt_config,
                local_config,
            )
            self.checkpoint = pretrained_model_name_or_path
            self.index_name = index_name
            self.config.experiment = "colbert"
            self.config.root = self.index_root

        if not training_mode:
            self.inference_ckpt = Checkpoint(
                self.checkpoint, colbert_config=self.config
            )
            self.base_model_max_tokens = (
                self.inference_ckpt.bert.config.max_position_embeddings
            ) - 4

        self.run_context = Run().context(self.run_config)
        self.run_context.__enter__()  # Manually enter the context
        self.searcher = None

    def _invert_pid_docid_map(self) -> Dict[str, List[int]]:
        d = defaultdict(list)
        for k, v in self.pid_docid_map.items():
            d[v].append(k)
        return d

    def _get_collection_files_from_disk(self, index_path: str):
        self.collection = srsly.read_json(index_path / "collection.json")
        if os.path.exists(str(index_path / "docid_metadata_map.json")):
            self.docid_metadata_map = srsly.read_json(
                str(index_path / "docid_metadata_map.json")
            )
        else:
            self.docid_metadata_map = None

        try:
            self.pid_docid_map = srsly.read_json(str(index_path / "pid_docid_map.json"))
        except FileNotFoundError as err:
            raise FileNotFoundError(
                "ERROR: Could not load pid_docid_map from index!",
                "This is likely because you are loading an older, incompatible index.",
            ) from err

        # convert all keys to int when loading from file because saving converts to str
        self.pid_docid_map = {
            int(key): value for key, value in self.pid_docid_map.items()
        }
        self.docid_pid_map = self._invert_pid_docid_map()

    def add_to_index(
        self,
        new_documents: List[str],
        new_pid_docid_map: Dict[int, str],
        new_docid_metadata_map: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        bsize: int = 32,
        use_faiss: bool = False,
    ):
        self.index_name = index_name if index_name is not None else self.index_name
        if self.index_name is None:
            print(
                "Cannot add to index without an index_name! Please provide one.",
                "Returning empty results.",
            )
            return None

        print(
            "WARNING: add_to_index support is currently experimental!",
            "add_to_index support will be more thorough in future versions",
        )

        if self.loaded_from_index:
            index_root = self.config.root
        else:
            expected_path_segment = Path(self.config.experiment) / "indexes"
            if str(expected_path_segment) in self.config.root:
                index_root = self.config.root
            else:
                index_root = str(Path(self.config.root) / expected_path_segment)

            if not self.collection:
                collection_path = Path(index_root) / self.index_name / "collection.json"
                if collection_path.exists():
                    self._get_collection_files_from_disk(
                        str(Path(index_root) / self.index_name)
                    )

        new_documents_with_ids = [
            {"content": doc, "document_id": new_pid_docid_map[pid]}
            for pid, doc in enumerate(new_documents)
            if new_pid_docid_map[pid] not in self.pid_docid_map.values()
        ]

        max_existing_pid = max(self.pid_docid_map.keys(), default=-1)
        for idx, doc in enumerate(new_documents_with_ids, start=max_existing_pid + 1):
            self.pid_docid_map[idx] = doc["document_id"]

        new_collection = [doc["content"] for doc in new_documents_with_ids]

        # TODO We may want to load an existing index here instead;
        #      For now require that either index() was called, or an existing one was loaded.
        assert self.model_index is not None

        # TODO We probably want to store some of this in the model_index directly.
        self.model_index.add(
            self.config,
            self.checkpoint,
            self.collection,
            index_root,
            self.index_name,
            new_collection,
            verbose=self.verbose != 0,
            bsize=bsize,
            use_faiss=use_faiss,
        )
        self.config = self.model_index.config

        # Update and serialize the index metadata + collection.
        self.collection = self.collection + new_collection

        # TODO This has inconsistent behavior for duplicates.
        if new_docid_metadata_map is not None:
            self.docid_metadata_map = self.docid_metadata_map or defaultdict(
                lambda: None
            )
            self.docid_metadata_map.update(new_docid_metadata_map)

        self.docid_pid_map = defaultdict(list)
        for pid, docid in self.pid_docid_map.items():
            self.docid_pid_map[docid].append(pid)

        self._save_index_metadata()

        print(
            f"Successfully updated index with {len(new_documents_with_ids)} new documents!\n",
            f"New index size: {len(self.collection)}",
        )

    def delete_from_index(
        self,
        document_ids: Union[TypeVar("T"), List[TypeVar("T")]],
        index_name: Optional[str] = None,
    ):
        self.index_name = index_name if index_name is not None else self.index_name
        if self.index_name is None:
            print(
                "Cannot delete from index without an index_name! Please provide one.",
                "Returning empty results.",
            )
            return None

        print(
            "WARNING: delete_from_index support is currently experimental!",
            "delete_from_index support will be more thorough in future versions",
        )

        pids_to_remove = []
        for pid, docid in self.pid_docid_map.items():
            if docid in document_ids:
                pids_to_remove.append(pid)

        # TODO We may want to load an existing index here instead;
        #      For now require that either index() was called, or an existing one was loaded.
        assert self.model_index is not None

        # TODO We probably want to store some of this in the model_index directly.
        self.model_index.delete(
            self.config,
            self.checkpoint,
            self.collection,
            self.index_name,
            pids_to_remove,
            verbose=self.verbose != 0,
        )

        # Update and serialize the index metadata + collection.
        self.collection = [
            doc for pid, doc in enumerate(self.collection) if pid not in pids_to_remove
        ]
        self.pid_docid_map = {
            pid: docid
            for pid, docid in self.pid_docid_map.items()
            if pid not in pids_to_remove
        }

        if self.docid_metadata_map is not None:
            self.docid_metadata_map = {
                docid: metadata
                for docid, metadata in self.docid_metadata_map.items()
                if docid not in document_ids
            }

        self._save_index_metadata()

        print(f"Successfully deleted documents with these IDs: {document_ids}")

    def _write_collection_files_to_disk(self):
        srsly.write_json(self.index_path + "/collection.json", self.collection)
        srsly.write_json(self.index_path + "/pid_docid_map.json", self.pid_docid_map)
        if self.docid_metadata_map is not None:
            srsly.write_json(
                self.index_path + "/docid_metadata_map.json", self.docid_metadata_map
            )

        # update the in-memory inverted map every time the files are saved to disk
        self.docid_pid_map = self._invert_pid_docid_map()

    def _save_index_metadata(self):
        assert self.model_index is not None

        model_metadata = srsly.read_json(self.index_path + "/metadata.json")
        index_config = self.model_index.export_metadata()
        index_config["index_name"] = self.index_name
        # Ensure that the additional metadata we store does not collide with anything else.
        model_metadata["RAGatouille"] = {"index_config": index_config}  # type: ignore
        srsly.write_json(self.index_path + "/metadata.json", model_metadata)
        self._write_collection_files_to_disk()

    def index(
        self,
        collection: List[str],
        pid_docid_map: Dict[int, str],
        docid_metadata_map: Optional[dict] = None,
        index_name: Optional["str"] = None,
        max_document_length: int = 256,
        overwrite: Union[bool, str] = "reuse",
        bsize: int = 32,
        use_faiss: bool = False,
    ):
        self.collection = collection
        self.config.doc_maxlen = max_document_length

        if index_name is not None:
            if self.index_name is not None:
                print(
                    f"New index_name received!",
                    f"Updating current index_name ({self.index_name}) to {index_name}",
                )
            self.index_name = index_name
        else:
            if self.index_name is None:
                print(
                    f"No index_name received!",
                    f"Using default index_name ({self.checkpoint}_new_index)",
                )
            self.index_name = self.checkpoint + "new_index"

        self.index_path = str(
            Path(self.run_config.root)
            / Path(self.run_config.experiment)
            / "indexes"
            / self.index_name
        )
        self.config.root = str(
            Path(self.run_config.root) / Path(self.run_config.experiment) / "indexes"
        )

        self.pid_docid_map = pid_docid_map

        # inverted mapping for returning full docs
        self.docid_pid_map = defaultdict(list)
        for pid, docid in self.pid_docid_map.items():
            self.docid_pid_map[docid].append(pid)

        self.docid_metadata_map = docid_metadata_map

        self.model_index = ModelIndexFactory.construct(
            "PLAID",
            self.config,
            self.checkpoint,
            self.collection,
            self.index_name,
            overwrite,
            verbose=self.verbose != 0,
            bsize=bsize,
            use_faiss=use_faiss,
        )
        self.config = self.model_index.config
        self._save_index_metadata()

        print("Done indexing!")

        return self.index_path

    def search(
        self,
        query: Union[str, list[str]],
        index_name: Optional["str"] = None,
        k: int = 10,
        force_fast: bool = False,
        zero_index_ranks: bool = False,
        doc_ids: Optional[List[str]] = None,
    ):
        pids = None
        if doc_ids is not None:
            pids = []
            for doc_id in doc_ids:
                pids.extend(self.docid_pid_map[doc_id])

        force_reload = index_name is not None and index_name != self.index_name
        if index_name is not None:
            if self.index_name is not None and self.index_name != index_name:
                print(
                    f"New index_name received!",
                    f"Updating current index_name ({self.index_name}) to {index_name}",
                )
            self.index_name = index_name
        else:
            if self.index_name is None:
                print(
                    "Cannot search without an index_name! Please provide one.",
                    "Returning empty results.",
                )
                return None

        # TODO We may want to load an existing index here instead;
        #      For now require that either index() was called, or an existing one was loaded.
        assert self.model_index is not None

        results = self.model_index.search(
            self.config,
            self.checkpoint,
            self.collection,
            self.index_name,
            self.base_model_max_tokens,
            query,
            k,
            pids,
            force_reload,
            force_fast=force_fast,
        )

        to_return = []
        for result in results:
            result_for_query = []
            for id_, rank, score in zip(*result):
                document_id = self.pid_docid_map[id_]
                result_dict = {
                    "content": self.collection[id_],
                    "score": score,
                    "rank": rank - 1 if zero_index_ranks else rank,
                    "document_id": document_id,
                    "passage_id": id_,
                }

                if self.docid_metadata_map is not None:
                    if document_id in self.docid_metadata_map:
                        doc_metadata = self.docid_metadata_map[document_id]
                        result_dict["document_metadata"] = doc_metadata

                result_for_query.append(result_dict)

            to_return.append(result_for_query)

        if len(to_return) == 1:
            return to_return[0]
        return to_return

    def _search(self, query: str, k: int, pids: Optional[List[int]] = None):
        assert self.model_index is not None
        return self.model_index._search(query, k, pids)

    def _batch_search(self, query: list[str], k: int):
        assert self.model_index is not None
        return self.model_index._batch_search(query, k)

    def train(self, data_dir, training_config: ColBERTConfig):
        training_config = ColBERTConfig.from_existing(self.config, training_config)
        training_config.nway = 2
        with Run().context(self.run_config):
            trainer = Trainer(
                triples=str(data_dir / "triples.train.colbert.jsonl"),
                queries=str(data_dir / "queries.train.colbert.tsv"),
                collection=str(data_dir / "corpus.train.colbert.tsv"),
                config=training_config,
            )

            trainer.train(checkpoint=self.checkpoint)

    def _colbert_score(self, Q, D_padded, D_mask):
        if ColBERTConfig().total_visible_gpus > 0:
            Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

        assert Q.dim() == 3, Q.size()
        assert D_padded.dim() == 3, D_padded.size()
        assert Q.size(0) in [1, D_padded.size(0)]

        scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
        scores = scores.max(1).values
        return scores.sum(-1)

    def _index_free_search(
        self,
        embedded_queries,
        documents: list[str],
        embedded_docs,
        doc_mask,
        k: int = 10,
        zero_index: bool = False,
    ):
        results = []

        for query in embedded_queries:
            results_for_query = []
            scores = self._colbert_score(query, embedded_docs, doc_mask)
            sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            high_score_idxes = [index for index, _ in sorted_scores[:k]]
            for rank, doc_idx in enumerate(high_score_idxes):
                result = {
                    "content": documents[doc_idx],
                    "score": float(scores[doc_idx]),
                    "rank": rank - 1 if zero_index else rank,
                    "result_index": doc_idx,
                }
                results_for_query.append(result)
            results.append(results_for_query)

        if len(results) == 1:
            return results[0]

        return results

    def _set_inference_max_tokens(
        self, documents: list[str], max_tokens: Union[Literal["auto"], int] = "auto"
    ):
        if (
            not hasattr(self, "inference_ckpt_len_set")
            or self.inference_ckpt_len_set is False
        ):
            if max_tokens == "auto":
                max_tokens = self.base_model_max_tokens
            else:
                max_tokens = int(max_tokens)
            if max_tokens > self.base_model_max_tokens:
                max_tokens = self.base_model_max_tokens
                percentile_90 = np.percentile(
                    [len(x.split(" ")) for x in documents], 90
                )
                max_tokens = min(
                    math.floor((math.ceil((percentile_90 * 1.35) / 32) * 32) * 1.1),
                    self.base_model_max_tokens,
                )
                max_tokens = max(256, max_tokens)

                if max_tokens > 300:
                    print(
                        f"Your documents are roughly {percentile_90} tokens long at the 90th percentile!",
                        "This is quite long and might slow down reranking!\n",
                        "Provide fewer documents, build smaller chunks or run on GPU",
                        "if it takes too long for your needs!",
                    )
            self.inference_ckpt.colbert_config.max_doclen = max_tokens
            self.inference_ckpt.doc_tokenizer.doc_maxlen = max_tokens
            self.inference_ckpt_len_set = True

    def _index_free_retrieve(
        self,
        query: Union[str, list[str]],
        documents: list[str],
        k: int,
        max_tokens: Union[Literal["auto"], int] = "auto",
        zero_index: bool = False,
        bsize: Union[Literal["auto"], int] = "auto",
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)

        if k > len(documents):
            print("k value cannot be larger than the number of documents! aborting...")
            return None
        if len(documents) > 1000:
            print(
                "Please note ranking in-memory is not optimised for large document counts! ",
                "Consider building an index and using search instead!",
            )
        if len(set(documents)) != len(documents):
            print(
                "WARNING! Your documents have duplicate entries! ",
                "This will slow down calculation and may yield subpar results",
            )

        embedded_queries = self._encode_index_free_queries(query, bsize=bsize)
        embedded_docs, doc_mask = self._encode_index_free_documents(
            documents, bsize=bsize
        )

        return self._index_free_search(
            embedded_queries=embedded_queries,
            documents=documents,
            embedded_docs=embedded_docs,
            doc_mask=doc_mask,
            k=k,
            zero_index=zero_index,
        )

    def _encode_index_free_queries(
        self,
        queries: Union[str, list[str]],
        bsize: Union[Literal["auto"], int] = "auto",
    ):
        if bsize == "auto":
            bsize = 32
        if isinstance(queries, str):
            queries = [queries]
        maxlen = max([int(len(x.split(" ")) * 1.35) for x in queries])
        self.inference_ckpt.query_tokenizer.query_maxlen = max(
            min(maxlen, self.base_model_max_tokens), 32
        )
        embedded_queries = [
            x.unsqueeze(0)
            for x in self.inference_ckpt.queryFromText(queries, bsize=bsize)
        ]
        return embedded_queries

    def _encode_index_free_documents(
        self,
        documents: list[str],
        bsize: Union[Literal["auto"], int] = "auto",
        verbose: bool = True,
    ):
        if bsize == "auto":
            bsize = 32
            if self.inference_ckpt.doc_tokenizer.doc_maxlen > 512:
                bsize = max(
                    1,
                    int(
                        32
                        / (
                            2
                            ** round(
                                math.log(
                                    self.inference_ckpt.doc_tokenizer.doc_maxlen, 2
                                )
                            )
                            / 512
                        )
                    ),
                )
                print("BSIZE:")
                print(bsize)
        embedded_docs = self.inference_ckpt.docFromText(
            documents, bsize=bsize, showprogress=verbose
        )[0]
        doc_mask = torch.full(embedded_docs.shape[:2], -float("inf")).to(
            embedded_docs.device
        )
        return embedded_docs, doc_mask

    def rank(
        self,
        query: str,
        documents: list[str],
        k: int = 10,
        zero_index_ranks: bool = False,
        bsize: int = 32,
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens="auto")
        self.inference_ckpt_len_set = False
        return self._index_free_retrieve(
            query, documents, k, zero_index=zero_index_ranks, bsize=bsize
        )

    def encode(
        self,
        documents: list[str],
        document_metadatas: Optional[list[dict]] = None,
        bsize: int = 32,
        max_tokens: Union[Literal["auto"], int] = "auto",
        verbose: bool = True,
    ):
        self._set_inference_max_tokens(documents=documents, max_tokens=max_tokens)
        encodings, doc_masks = self._encode_index_free_documents(
            documents, bsize=bsize, verbose=verbose
        )
        encodings = torch.cat(
            [
                encodings,
                torch.zeros(
                    (
                        encodings.shape[0],
                        self.inference_ckpt.doc_tokenizer.doc_maxlen
                        - encodings.shape[1],
                        encodings.shape[2],
                    )
                ).to(device=encodings.device),
            ],
            dim=1,
        )
        doc_masks = torch.cat(
            [
                doc_masks,
                torch.full(
                    (
                        doc_masks.shape[0],
                        self.inference_ckpt.colbert_config.max_doclen
                        - doc_masks.shape[1],
                    ),
                    -float("inf"),
                ).to(device=doc_masks.device),
            ],
            dim=1,
        )

        if verbose:
            print("Shapes:")
            print(f"encodings: {encodings.shape}")
            print(f"doc_masks: {doc_masks.shape}")

        if hasattr(self, "in_memory_collection"):
            if self.in_memory_metadata is not None:
                if document_metadatas is None:
                    self.in_memory_metadatas.extend([None] * len(documents))
                else:
                    self.in_memory_metadata.extend(document_metadatas)
            elif document_metadatas is not None:
                self.in_memory_metadata = [None] * len(self.in_memory_collection)
                self.in_memory_metadata.extend(document_metadatas)

            self.in_memory_collection.extend(documents)

            # add 0 padding to encodings so they're self.inference_ckpt.doc_tokenizer.doc_maxlen length

            self.in_memory_embed_docs = torch.cat(
                [self.in_memory_embed_docs, encodings], dim=0
            )
            self.doc_masks = torch.cat([self.doc_masks, doc_masks], dim=0)

        else:
            self.in_memory_collection = documents
            self.in_memory_metadata = document_metadatas
            self.in_memory_embed_docs = encodings
            self.doc_masks = doc_masks

    def search_encoded_docs(
        self,
        queries: Union[str, list[str]],
        k: int = 10,
        bsize: int = 32,
    ):
        queries = self._encode_index_free_queries(queries, bsize=bsize)
        results = self._index_free_search(
            embedded_queries=queries,
            documents=self.in_memory_collection,
            embedded_docs=self.in_memory_embed_docs,
            doc_mask=self.doc_masks,
            k=k,
        )
        if self.in_memory_metadata is not None:
            for result in results:
                result["document_metadata"] = self.in_memory_metadata[
                    result["result_index"]
                ]
        return results

    def clear_encoded_docs(self, force: bool = False):
        if not force:
            print(
                "All in-memory encodings will be deleted in 10 seconds, interrupt now if you want to keep them!"
            )
            print("...")
            time.sleep(10)
        del self.in_memory_collection
        del self.in_memory_metadata
        del self.in_memory_embed_docs
        del self.doc_masks
        del self.inference_ckpt_len_set

    def __del__(self):
        # Clean up context
        try:
            self.run_context.__exit__(None, None, None)
        except Exception:
            print("INFO: Tried to clean up context but failed!")

    def evaluate(
        self,
        queries: list[str],
        expected_document_ids: list[list[str]],
        expected_passage_ids: list[list[int]] = None,
        metrics: list[str] = None,
        k: list[int] = None,
    ) -> dict[str, dict[int, float]]:
        search_results = self.search(queries, k=max(k))
        if len(queries) == 1:
            search_results = [search_results]
        sorted_search_results = [
            sorted(results, key=lambda x: x["rank"]) for results in search_results
        ]
        retrieved_ids = [
            [
                (
                    (result["document_id"], result["passage_id"])
                    if expected_passage_ids
                    else result["document_id"]
                )
                for result in results
            ]
            for results in sorted_search_results
        ]

        metric_types = resolve_metrics(metrics)
        metric_instances = [metric() for metric in metric_types]

        if expected_passage_ids:
            expected_ids = [
                [
                    (doc_id, passage_id)
                    for doc_id, passage_id in zip(doc_ids, passage_ids)
                ]
                for doc_ids, passage_ids in zip(
                    expected_document_ids, expected_passage_ids
                )
            ]
        else:
            expected_ids = expected_document_ids

        metric_dict = {}
        for metric in metric_instances:
            metric_dict[metric.metric_name] = {}
            for k_ in k:
                eval_results = [
                    metric.compute(expected_ids_, retrieved_ids[i][:k_])
                    for i, expected_ids_ in enumerate(expected_ids)
                ]
                metric_dict[metric.metric_name][k_] = np.mean(eval_results)

        return metric_dict
