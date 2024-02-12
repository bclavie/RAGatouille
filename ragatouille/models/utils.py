import os
import random
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from colbert.infra import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoModel, BertPreTrainedModel


def seeded_shuffle(collection: list, seed: int = 42):
    random.seed(seed)
    random.shuffle(collection)
    return collection


""" HUGGINGFACE """


def export_to_huggingface_hub(
    colbert_path: Union[str, Path],
    huggingface_repo_name: str,
    # index_path: Optional[str] = None,
    export_vespa_onnx: bool = False,
    use_tmp_dir: bool = False,
):
    # TODO: If index_path is available, convert the index directory into Vespa Format (?) and push it too.

    # ensure model contains a valid ColBERT config before exporting
    colbert_config = ColBERTConfig.load_from_checkpoint(colbert_path)
    try:
        assert colbert_config is not None
    except Exception:
        print(f"Path {colbert_path} does not contain a valid ColBERT config!")

    export_path = colbert_path
    if use_tmp_dir:
        export_path = ".tmp/hugging_face_export"
        print("Using tmp dir to store export files...")
        colbert_model = ColBERT(
            colbert_path,
            colbert_config=colbert_config,
        )
        print(f"Model loaded... saving export files to disk at {export_path}")
        try:
            save_model = colbert_model.save
        except Exception:
            save_model = colbert_model.module.save
        save_model(export_path)

    if export_vespa_onnx:
        rust_tokenizer_available = True
        if use_tmp_dir:
            try:
                colbert_model.raw_tokenizer.save_pretrained(
                    export_path, legacy_format=False
                )
            except Exception:
                rust_tokenizer_available = False
        else:
            rust_tokenizer_available = os.path.exists(
                Path(colbert_path) / "tokenizer.json"
            )
        if not rust_tokenizer_available:
            print(
                "The tokenizer for your model does not seem to have a Fast Tokenizer implementation...\n",
                "This may cause problems when trying to use with Vespa!\n",
                "Proceeding anyway...",
            )

        export_to_vespa_onnx(colbert_path, out_path=export_path)
    try:
        api = HfApi()
        api.create_repo(repo_id=huggingface_repo_name, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=export_path,
            repo_id=huggingface_repo_name,
            repo_type="model",
        )
        print(f"Successfully uploaded model to {huggingface_repo_name}")
    except ValueError as e:
        print(
            f"Could not create repository on the huggingface hub.\n",
            f"Error: {e}\n",
            "Please make sure you are logged in (run huggingface-cli login)\n",
            "If the error persists, please open an issue on github. This is a beta feature!",
        )
    except HfHubHTTPError:
        print(
            "You don't seem to have the rights to create a repository with this name...\n",
            "Please make sure your repo name is in the format 'yourusername/your-repo-name'",
        )
    finally:
        if use_tmp_dir:
            shutil.rmtree(export_path)

def check_model_exists(model_name):
    api = HfApi()
    try:
        _ = api.model_info(model_name)
        return True
    except Exception as e:
        return False

def upload_index_and_model(
    colbert_path: Union[str, Path],
    huggingface_repo_name: str,
    index_path: Optional[Path] = None,
):
    # ensure model contains a valid ColBERT config before exporting
    colbert_config = ColBERTConfig.load_from_checkpoint(colbert_path)
    print(colbert_config)
    try:
        assert colbert_config is not None
    except Exception as e:
        print(f"Path {colbert_path} does not contain a valid ColBERT config!")
        raise ValueError from e

    export_path = ".tmp/hugging_face_export"
    print("Using tmp dir to store export files...")

    model_name = colbert_config.checkpoint
    print(model_name)
    if check_model_exists(model_name):
        print(f"The model {model_name} already exists on Huggingface Hub.")
    else:
        colbert_model = ColBERT(
            colbert_path,
            colbert_config=colbert_config,
        )
        print(f"Model loaded... saving export files to disk at {export_path}")
        try:
            save_model = colbert_model.save
        except Exception:
            save_model = colbert_model.module.save
        save_model(str(Path(export_path) / "model"))

        # TODO: before exporting the index, we'd need to update the Index's config (in metadata.json) to make sure checkpoint points to path/to/your/index/model, so it loads properly!
        # colbert_config.checkpoint = str(index_path)
        # colbert_config.save(Path(colbert_path) / "config.json", overwrite=True)

    if index_path is not None and Path(index_path).exists():
        index_export_path = Path(export_path)
        shutil.copytree(index_path, index_export_path, dirs_exist_ok=True)
        print(f"Index files copied to {index_export_path}")
    else:
        raise FileExistsError("No index_path provided or path does not exist.")

    try:
        api = HfApi()
        api.create_repo(repo_id=huggingface_repo_name, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=export_path,
            repo_id=huggingface_repo_name,
            repo_type="model",
        )
        print(f"Successfully uploaded model to {huggingface_repo_name}")
    except ValueError as e:
        print(
            f"Could not create repository on the huggingface hub.\n",
            f"Error: {e}\n",
            "Please make sure you are logged in (run huggingface-cli login)\n",
            "If the error persists, please open an issue on github. This is a beta feature!",
        )
    except HfHubHTTPError:
        print(
            "You don't seem to have the rights to create a repository with this name...\n",
            "Please make sure your repo name is in the format 'yourusername/your-repo-name'",
        )
    finally:
        shutil.rmtree(export_path)


""" VESPA """


class VespaColBERT(BertPreTrainedModel):
    def __init__(self, config, dim):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)


def export_to_vespa_onnx(
    colbert_path: Union[str, Path],
    out_path: Union[str, Path],
    out_file_name: str = "vespa_colbert.onnx",
):
    print(f"Exporting model {colbert_path} to {out_path}/{out_file_name}")
    out_path = Path(out_path)
    vespa_colbert = VespaColBERT.from_pretrained(colbert_path, dim=128)
    print("Model loaded, converting to ONNX...")
    input_names = ["input_ids", "attention_mask"]
    output_names = ["contextual"]
    input_ids = torch.ones(1, 32, dtype=torch.int64)
    attention_mask = torch.ones(1, 32, dtype=torch.int64)
    args = (input_ids, attention_mask)
    torch.onnx.export(
        vespa_colbert,
        args=args,
        f=str(out_path / out_file_name),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "batch"},
            "attention_mask": {0: "batch", 1: "batch"},
            "contextual": {0: "batch", 1: "batch"},
        },
        opset_version=17,
    )
    print("Vespa ONNX export complete!")
