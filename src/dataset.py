from dataclasses import dataclass, field
import logging
import sys
import typing

import datasets
from transformers import PreTrainedTokenizerBase

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_csv_dataset(
    path: str,
    tokenizer: typing.Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 256,
    pad_to_max_length: bool = True,
    overwrite_cache: bool = False,
) -> datasets.Dataset:
    # set a padding option
    padding = "max_length" if pad_to_max_length else False

    # set a sequence length option
    model_max_length = tokenizer.model_max_length
    if max_length > model_max_length:
        logger.warning(
            f"The max_length passed ({max_length}) is larger than the maximum length for the"
            f"model ({model_max_length}). Using max_length={model_max_length}."
        )
    max_length = min(max_length, model_max_length)

    dataset: datasets.Dataset = datasets.load_dataset(
        "csv",
        data_files=path,
        split=datasets.Split.TRAIN, # set to return `datasets.Dataset`
    )

    # applay the tokenizer
    if tokenizer is not None:
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                padding=padding,
                max_length=max_length,
                truncation=True,
            ),
            batched=True,
            load_from_cache_file=not (overwrite_cache),
            desc="Running tokenizer on dataset",
        )

    # encode label to id
    dataset = dataset.class_encode_column("label")

    return dataset
