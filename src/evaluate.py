from dataclasses import dataclass, field
import logging
import sys
import os
import typing

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from dataset import load_csv_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class CommonArguments:
    # dataset
    test_path: typing.Optional[str] = field(
        default=None,
        metadata={"help": "A csv path containing the validation data."},
    )
    save_items_dir: typing.Optional[str] = field(
        default=None,
        metadata={"help": "A csv path containing the validation data."},
    )
    max_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    cache_dir: typing.Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def compute_metrics(p: EvalPrediction) -> typing.Optional[typing.Dict[str, float]]:
    logits = p.predictions
    labels = p.label_ids
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"acc": acc}


def main():
    parser = HfArgumentParser((CommonArguments, TrainingArguments))
    (common_args, training_args) = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        common_args.save_items_dir,
        local_files_only=True
    )

    # load datasets
    test_dataset = load_csv_dataset(
        common_args.test_path,
        tokenizer,
        common_args.max_length,
        common_args.pad_to_max_length,
    )

    # load transformer model
    model = BertForSequenceClassification.from_pretrained(
        os.path.join(common_args.save_items_dir, "pytorch_model.bin"),
        config=os.path.join(common_args.save_items_dir, "config.json"),
        local_files_only=True
    )
    model.to(device)
    model.eval()

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # start evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
