from pathlib import Path
from typing import Literal, Optional, Union

from colbert.infra import ColBERTConfig

from ragatouille.data import TrainingDataProcessor
from ragatouille.models import ColBERT, LateInteractionModel
from ragatouille.negative_miners import HardNegativeMiner, SimpleMiner
from ragatouille.utils import seeded_shuffle


class RAGTrainer:
    """Main trainer to fine-tune/train ColBERT models with a few lines."""

    def __init__(
        self,
        model_name: str,
        pretrained_model_name: str,
        language_code: str = "en",
        n_usable_gpus: int = -1,
    ):
        """
        Initialise a RAGTrainer instance. This will load a base model: either an existing ColBERT model to fine-tune or a BERT/RoBERTa-like model to build a new ColBERT model from.

        Parameters:
            model_name: str - Name of the model to train. This will be used to name the checkpoints and the index.
            pretrained_model_name: str - Name of the pretrained model to use as a base. Can be a local path to a checkpoint or a huggingface model name.
            language_code: str - Language code of the model to train. This will be used to name the checkpoints and the index.
            n_usable_gpus: int - Number of GPUs to use. By default, value is -1, which means use all available GPUs or none if no GPU is available.

        Returns:
            self (RAGTrainer): The current instance of RAGTrainer, with the base model initialised.
        """
        self.model_name = model_name
        self.pretrained_model_name = pretrained_model_name
        self.language_code = language_code
        self.model: Union[LateInteractionModel, None] = ColBERT(
            pretrained_model_name_or_path=pretrained_model_name,
            n_gpu=n_usable_gpus,
            training_mode=True,
        )
        self.negative_miner: Union[HardNegativeMiner, None] = None
        self.collection: list[str] = []
        self.queries: Union[list[str], None] = None
        self.raw_data: Union[list[tuple], list[list], None] = None
        self.training_triplets: list[list[int]] = list()

    def add_documents(self, documents: list[str]):
        self.collection += documents
        seeded_shuffle(self.collection)

    def export_training_data(self, path: Union[str, Path]):
        """
        Manually export the training data processed by prepare_training_data to a given path.

        Parameters:
            path: Union[str, Path] - Path to the directory where the data will be exported."""
        self.data_processor.export_training_data(path)

    def _add_to_collection(self, content: Union[str, list, dict]):
        if isinstance(content, str):
            self.collection.append(content)
        elif isinstance(content, list):
            self.collection += [txt for txt in content if isinstance(txt, str)]
        elif isinstance(content, dict):
            self.collection += [content["content"]]

    def prepare_training_data(
        self,
        raw_data: Union[list[tuple], list[list]],
        all_documents: Optional[list[str]] = None,
        data_out_path: Union[str, Path] = "./data/",
        num_new_negatives: int = 10,
        hard_negative_minimum_rank: int = 10,
        mine_hard_negatives: bool = True,
        hard_negative_model_size: str = "small",
        pairs_with_labels: bool = False,
        positive_label: Union[int, str] = 1,
        negative_label: Union[int, str] = 0,
    ) -> str:
        """
        Fully pre-process input-data in various raw formats into ColBERT-ready files and triplets.
        Will accept a variety of formats, such as unannotated pairs, annotated pairs, triplets of strings and triplets of list of strings.
        Will process into a ColBERT-ready format and export to data_out_path.
        Will generate hard negatives if mine_hard_negatives is True.
        num_new_negatives decides how many negatives will be generated. if mine_hard_negatives is False and num_new_negatives is > 0, these negatives will be randomly sampled.

        Parameters:
            raw_data: Union[list[tuple], list[list]] - List of pairs, annotated pairs, or triplets of strings.
            all_documents: Optional[list[str]] - A corpus of documents to be used for sampling negatives.
            data_out_path: Union[str, Path] - Path to the directory where the data will be exported (can be a tmp directory).
            num_new_negatives: int - Number of new negatives to generate for each query.
            mine_hard_negatives: bool - Whether to use hard negatives mining or not.
            hard_negative_model_size: str - Size of the model to use for hard negatives mining.
            pairs_with_labels: bool - Whether the raw_data is a list of pairs with labels or not.
            positive_label: Union[int, str] - Label to use for positive pairs.
            negative_label: Union[int, str] - Label to use for negative pairs.

        Returns:
            data_out_path: Union[str, Path] - Path to the directory where the data has been exported.
        """
        if all_documents is not None:
            self.collection += [doc for doc in all_documents if isinstance(doc, str)]
        print(1)
        self.data_dir = Path(data_out_path)
        sample = raw_data[0]
        if len(sample) == 2:
            data_type = "pairs"
        elif len(sample) == 3:
            if pairs_with_labels:
                data_type = "labeled_pairs"
                if sample[-1] not in [positive_label, negative_label]:
                    raise ValueError(f"Invalid value for label: {sample}")
            else:
                data_type = "triplets"
        else:
            raise ValueError("Raw data must be a list of pairs or triplets of strings.")
        print(2)
        self.queries = set()
        for x in raw_data:
            if isinstance(x[0], str):
                self.queries.add(x[0])
            else:
                raise ValueError("Queries must be a strings.")

            self._add_to_collection(x[1])

            if data_type == "triplets":
                self._add_to_collection(x[2])

        self.collection = list(set(self.collection))
        seeded_shuffle(self.collection)
        print(3)
        if mine_hard_negatives:
            self.negative_miner = SimpleMiner(
                language_code=self.language_code,
                model_size=hard_negative_model_size,
            )
            self.negative_miner.build_index(self.collection)
        print(4)
        self.data_processor = TrainingDataProcessor(
            collection=self.collection,
            queries=self.queries,
            negative_miner=self.negative_miner if mine_hard_negatives else None,
        )
        print(5)
        self.data_processor.process_raw_data(
            data_type=data_type,
            raw_data=raw_data,
            export=True,
            data_dir=data_out_path,
            num_new_negatives=num_new_negatives,
            positive_label=positive_label,
            negative_label=negative_label,
            mine_hard_negatives=mine_hard_negatives,
            hard_negative_minimum_rank=hard_negative_minimum_rank,
        )
        print(6)
        if len(self.data_processor.training_triplets) == 0:
            if mine_hard_negatives:
                print(
                    "Warning: No training triplets were generated with setting mine_hard_negatives=='True'. This may be due to the data being too small or the hard negative miner not being able to find enough hard negatives."
                )
                self.data_processor.process_raw_data(
                    data_type=data_type,
                    raw_data=raw_data,
                    export=True,
                    data_dir=data_out_path,
                    num_new_negatives=num_new_negatives,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    mine_hard_negatives=False,
                    hard_negative_minimum_rank=hard_negative_minimum_rank,
                )
            else:
                raise ValueError("No training triplets were generated.")
        print(7)
        self.training_triplets = self.data_processor.training_triplets

        return data_out_path

    def train(
        self,
        batch_size: int = 32,
        nbits: int = 2,
        maxsteps: int = 500_000,
        use_ib_negatives: bool = True,
        learning_rate: float = 5e-6,
        dim: int = 128,
        doc_maxlen: int = 256,
        use_relu: bool = False,
        warmup_steps: Union[int, Literal["auto"]] = "auto",
        accumsteps: int = 1,
    ) -> str:
        """
        Launch training or fine-tuning of a ColBERT model.

        Parameters:
            batch_size: int - Total batch size -- divice by n_usable_gpus for per-GPU batch size.
            nbits: int - number of bits used for vector compression by the traiened model. 2 is usually ideal.
            maxsteps: int - End training early after maxsteps steps.
            use_ib_negatives: bool - Whether to use in-batch negatives to calculate loss or not.
            learning_rate: float - ColBERT litterature usually has this performing best between 3e-6 - 2e-5 depending on data size
            dim: int - Size of individual vector representations.
            doc_maxlen: int - The maximum length after which passages will be truncated
            warmup_steps: Union[int, Literal["auto"]] - How many warmup steps to use for the learning rate.
                                                      Auto will default to 10% of total steps
            accumsteps: How many gradient accummulation steps to use to simulate higher batch sizes.

        Returns:
            model_path: str - Path to the trained model.
        """
        if not self.training_triplets:
            total_triplets = sum(
                1 for _ in open(str(self.data_dir / "triples.train.colbert.jsonl"), "r")
            )
        else:
            total_triplets = len(self.training_triplets)

        training_config = ColBERTConfig(
            bsize=batch_size,
            model_name=self.model_name,
            name=self.model_name,
            checkpoint=self.pretrained_model_name,
            use_ib_negatives=use_ib_negatives,
            maxsteps=maxsteps,
            nbits=nbits,
            lr=learning_rate,
            dim=dim,
            doc_maxlen=doc_maxlen,
            relu=use_relu,
            accumsteps=accumsteps,
            warmup=int(total_triplets // batch_size * 0.1)
            if warmup_steps == "auto"
            else warmup_steps,
            save_every=int(total_triplets // batch_size // 10),
        )

        return self.model.train(data_dir=self.data_dir, training_config=training_config)
