from dataclasses import dataclass, field
import logging
import sys
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
    train_path: typing.Optional[str] = field(
        default=None,
        metadata={"help": "A csv path containing the training data."},
    )
    valid_path: typing.Optional[str] = field(
        default=None,
        metadata={"help": "A csv path containing the validation data."},
    )
    model_name_or_path: typing.Optional[str] = field(
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
    pre, rec, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        "acc": acc,
        "prcision": pre,
        "recall": rec,
        "fscore": f1,
    }


def main():
    parser = HfArgumentParser((CommonArguments, TrainingArguments))
    (common_args, training_args) = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load transformers tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        common_args.model_name_or_path,
        cache_dir=common_args.cache_dir,
        use_fast=common_args.use_fast_tokenizer,
        revision=common_args.model_revision,
        use_auth_token=(True if common_args.use_auth_token else None),
    )

    # load datasets
    train_dataset = load_csv_dataset(
        common_args.train_path,
        tokenizer,
        common_args.max_length,
        common_args.pad_to_max_length,
    ).shuffle(seed=training_args.seed)
    valid_dataset = load_csv_dataset(
        common_args.valid_path,
        tokenizer,
        common_args.max_length,
        common_args.pad_to_max_length,
    )
    n_label = len(dict.fromkeys(train_dataset["label"]))

    # load transformer model
    model = BertForSequenceClassification.from_pretrained(
        common_args.model_name_or_path,
        num_labels=n_label,
    )
    # froze the parameters of pretrained model
    for param in model.base_model.parameters():
        param.requires_grad = False
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
    )

    # start trainining
    result = trainer.train()
    metrics = result.metrics
    trainer.save_model()  # saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # start evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
