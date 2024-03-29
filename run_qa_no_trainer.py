#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import csv
import pandas as pd
import logging
import math
import os
import copy
import random
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from dataclasses import dataclass
import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, PaddingStrategy
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



def write_csv(csv_out_path, pred_result: List[Dict]):
    
    f = csv.writer(open(csv_out_path, "w", newline=''))

    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["id", "answer"])

    for item in pred_result:
        f.writerow([item["id"], item["prediction_text"]])
        
    print(f"Save {csv_out_path} successfully.")



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    
    # dataset 
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--context_file", type=str, default=None,
        help=(
            "A csv or a json file containing the context data." # from PC
            "used to map paragrph_numbers to content_string"
        )
    )
    
    # sample limit
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    
    # model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    
    # tokenizer
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--use_slow_tokenizer",
    #     action="store_true",
    #     help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    # )
    
    # sentence preprocessing
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="How many worker can do preprocessing."
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    
    # batch_size
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    
    # weight and gradient
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    # train_loop
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    
    # learning_rate
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    # training mode and log
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    # predition
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    
    # squad version2 -> if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along this flag
    parser.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=False,
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`.",
    )
    
    # save path
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--prediction_csv_dir", type=str, default=None, help="Where to store the predict result in a csv file.")
    
    # push to hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
        
    if args.prediction_csv_dir.split(".")[-1] != "csv":
        raise ValueError("File extension must be csv.")

    return args


@dataclass
class train_DataCollatorWithPadding (DataCollatorWithPadding):
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        # input_features ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'] & 
        #                ['id', 'context', 'question', 'answers', 'offset_mapping', 'example_id']
        
        # example ['id', 'context', 'question', 'answers']
        # dataset ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id']
        
        # features -> batch ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
        
        # remove column not correspond to post_processing_function() args
        batch_examples = copy.deepcopy(features)
        # print("="*100, "\n", type(batch_examples), type(batch_examples[0]), batch_examples[0].keys(), "\n")
        for dict in batch_examples:
            dict.pop('input_ids')
            dict.pop('token_type_ids')
            dict.pop('attention_mask')
            dict.pop('start_positions')
            dict.pop('end_positions')
            dict.pop('offset_mapping')
            dict.pop('example_id')
        # print("="*100, "\n", type(batch_examples), type(batch_examples[0]), batch_examples[0].keys(), "\n")
        df_batch_examples = pd.DataFrame(batch_examples)
        # print("="*100, "\n", type(df_batch_examples), df_batch_examples, "\n")
        ApacheArrow_batch_examples = Dataset.from_pandas(df_batch_examples)
        # print("="*100, "\n", type(ApacheArrow_batch_examples), ApacheArrow_batch_examples[0], "\n")
        # input("=> In train_DataCollatorWithPadding, print(batch_examples), press Any key to continue")
        
        
        # remove column not correspond to post_processing_function() args
        batch_dataset = copy.deepcopy(features)
        for dict in batch_dataset:
            dict.pop('start_positions')
            dict.pop('end_positions')
            dict.pop('id')
            dict.pop('context')
            dict.pop('question')
            dict.pop('answers')
        df_batch_dataset = pd.DataFrame(batch_dataset)
        ApacheArrow_batch_dataset = Dataset.from_pandas(df_batch_dataset)
        
        
        # remove column not correspond to batch return here
        for dict in features:
            dict.pop('id')
            dict.pop('context')
            dict.pop('question')
            dict.pop('answers')
            dict.pop('offset_mapping')
            dict.pop('example_id')
        
        # 原始 sample code 做的處理
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # print("="*100, "\n", features, "\n")
        # print(batch, "\n")
        # input("=> In DataCollatorWithPadding(), press Any key to continue")
        
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        

        # print("="*100, "\n", f"batch = {batch}\n")
        # print("="*100, "\n", f"ApacheArrow_batch_examples = {ApacheArrow_batch_examples[0]}\n")
        # print("="*100, "\n", f"ApacheArrow_batch_dataset = {ApacheArrow_batch_dataset[0]}\n")
        # input("=> In Dataloader process, press Any key to continue")

        return batch, ApacheArrow_batch_examples, ApacheArrow_batch_dataset



def dataset_change_format(data_path:str, context_path:str) -> pd.DataFrame:
    
    print("="*100, "\n", f"=> Apply dataset_change_format() on '{data_path}'\n")
    
    context_mapping = json.loads(Path(context_path).read_text())
    data_Dict = json.loads(Path(data_path).read_text())
    
    print(f"Before change: len = {len(data_Dict)}, content_format = \n{data_Dict[0]}\n")
    data_List = list()
    for i in range(len(data_Dict)):
        data_List.append({"id": data_Dict[i]["id"],
                          "context":context_mapping[data_Dict[i]["relevant"]],
                          "question":data_Dict[i]["question"],
                          "answers":{
                              "text":[ data_Dict[i]["answer"]["text"], ],
                              "answer_start":[ data_Dict[i]["answer"]["start"], ],
                          }
                        })
    print(f"After change: len = {len(data_List)}, content_format = \n{data_List[0]}\n")
    return pd.DataFrame(data_List)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    torch.cuda.set_device(1)
    print("=> torch.cuda.set_device(1)")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        # data_files = {}
        if args.train_file is not None:
            df = dataset_change_format(args.train_file, args.context_file)
            raw_datasets_train = Dataset.from_pandas(df)
            # data_files["train"] = args.train_file
        if args.validation_file is not None:
            df = dataset_change_format(args.validation_file, args.context_file)
            raw_datasets_valid = Dataset.from_pandas(df)
            # data_files["validation"] = args.validation_file
        # if args.test_file is not None:
        #     # data_files["test"] = args.test_file
        raw_datasets = datasets.DatasetDict({"train": raw_datasets_train, "validation": raw_datasets_valid})
        # extension = args.train_file.split(".")[-1]
        # raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    
    # # Testing: dataset_change_format
    # df_funcTest = pd.read_json(args.test_file) # Load "QA_sheet.json" as "args.test_file"
    # funcTest = Dataset.from_pandas(df_funcTest)
    # print("="*100, "\n", type(funcTest), type(funcTest[0]))
    # print(f"funcTest.column_names = {funcTest.column_names}\n")
    # print(f"funcTest[0] = {funcTest[0]}\n")
    # input("=> Testing: dataset_change_format, press Any key to continue")
    
    print("="*100, "\n", type(raw_datasets), type(raw_datasets['train']), type(raw_datasets['train'][0]), "\n")
    print(f"raw_datasets.column_names = {raw_datasets.column_names}\n")
    print(f"raw_datasets['train'][0] = {raw_datasets['train'][0]}\n")
    # input("=> Load raw_datasets (default dataset), press Any key to continue")
    

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)

    # Preprocessing the datasets.
    # Preprocessing is slightly different for training and evaluation.

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length) # 最大 seq_length只能是 tokenizer max 因為超過就不在 tokenizer 的 train domain 了

    # Training preprocessing function
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]] # lstrip() 方法用於截掉字符串左邊的空格或指定字符。

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        tokenized_examples["example_id"] = []
        tokenized_examples['id'] = []
        tokenized_examples['context'] = []
        tokenized_examples['question'] = []
        tokenized_examples['answers'] = []
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["id"].append(examples["id"][sample_index])
            tokenized_examples["context"].append(examples["context"][sample_index])
            tokenized_examples["question"].append(examples["question"][sample_index])
            tokenized_examples["answers"].append(examples["answers"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        # assert len(offset_mapping) == len(tokenized_examples["input_ids"]), "ERROR: len(offset_mapping) != len(tokenized_examples['input_ids']"
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    # end of "Training preprocessing" function
    
    # Train dataset extraction and Preprocessing
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset") # --do_train 應該是 with_trainer 的 args，雖然 no_trainer 版本內沒有這個 args 但描述就順用
    train_examples = raw_datasets["train"] # <class 'datasets.arrow_dataset.Dataset'>
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_examples = train_examples.select(range(args.max_train_samples))
    # Create train feature from dataset
    with accelerator.main_process_first():
        train_dataset = train_examples.map(
            prepare_train_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        if args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))

    print("="*100, "\n", type(train_examples), "\n") #  train_examples = 未前處理 # <class 'datasets.arrow_dataset.Dataset'>
    print(train_examples.column_names, "\n") # ['id', 'context', 'question', 'answers']
    print(train_examples[0], "\n\n")
    print(type(train_dataset), "\n") # train_dataset = 已前處理 # <class 'datasets.arrow_dataset.Dataset'>
    print(train_dataset.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions'],
                                            # ['id', 'context', 'question', 'answers', 'offset_mapping', 'example_id']
    print(train_dataset[0], "\n")
    # input("=> Train dataset extraction and Preprocessing complete, press Any key to continue")

    # Validation preprocessing function
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # 不 pop("offset_mapping")
        
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    # end of "Validation preprocessing" function
    
    # Validation dataset extraction and Preprocessing
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset") # --do_eval 應該是 with_trainer 的 args，雖然 no_trainer 版本內沒有這個 args 但描述就順用
    eval_examples = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    # Validation Feature Creation
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    print("="*100, "\n", type(eval_examples), "\n") # eval_examples = 未前處理 # <class 'datasets.arrow_dataset.Dataset'>
    print(eval_examples.column_names, "\n") # ['id', 'context', 'question', 'answers']
    print(eval_examples[0], "\n\n")
    print(type(eval_dataset), "\n") # eval_dataset = 已前處理 # <class 'datasets.arrow_dataset.Dataset'>
    print(eval_dataset.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id']
    print(eval_dataset[0], "\n")
    # input("=> Validation dataset extraction and Preprocessing complete, press Any key to continue")
    
    # Test dataset extraction and Preprocessing
    # NOTE: Predict is treated as validation
    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
            
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            if args.max_predict_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_examples)), 3):
        logger.info(f"\nSample {index} of the train_examples: {train_examples[index]}.\n")
        logger.info(f"\nSample {index} of the train_dataset: {train_dataset[index]}.\n")
    # input("=> Log a few random samples from the training set, press Any key to continue")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
        train_data_collator = train_DataCollatorWithPadding
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        train_data_collator = train_DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        

    # train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=args.per_device_train_batch_size
    )
    # print("="*100, "\n", type(train_dataset_for_model), "\n")
    # print(train_dataset_for_model.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
    print("="*100, "\n", type(train_dataset), "\n")
    print(train_dataset.column_names, "\n") # ['id', 'context', 'question', 'answers'], 
                                            # ['input_ids', 'token_type_ids', 'attention_mask'], 
                                            # ['offset_mapping', 'example_id', 'start_positions', 'end_positions']
    # input("=> Train DataLoader ready, press Any key to continue")

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    print("="*100, "\n", type(eval_dataset_for_model), "\n")
    print(eval_dataset_for_model.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask']
    # input("=> Validation DataLoader ready, press Any key to continue")

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            null_score_diff_threshold=args.null_score_diff_threshold,
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        
        references = []
        stored_id = []
        if len(formatted_predictions) != len(examples):
            for i in range(len(examples)):
                if examples[i]["id"] in stored_id:
                    continue
                references.append({"id": examples[i]["id"], "answers": examples[i][answer_column_name]})
                stored_id.append(examples[i]["id"])
                print(references, stored_id)
                # input("-> In post_processing_function(), create references, press Any key to continue")
        else:
            references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
            

        # print(f"=> After post_processing_function(),\n", formatted_predictions, "\n", references, "\n")
        # input("-> In post_processing_function(), print(references), press Any key to continue")
        
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)
    # print(f"model.named_parameters = \n{model.named_parameters}")
    # input(f"=> Optimizer and Move to accelerator_device {device}, press Any key to continue")

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"): # hasattr() 函數用於判斷對像是否包含對應的屬性。
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    # if args.with_tracking:
    #     experiment_config = vars(args)
    #     # TensorBoard cannot log Enums, need the raw value
    #     experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    #     accelerator.init_trackers("qa_no_trainer", experiment_config)

    # Metrics
    metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")
    print("="*100, "\n", metric, "\n")
    # input("=> Load metric, press Any key to continue")
    
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Train counter, logs, parameters
    completed_steps = 0 # only for train, because of using "gradient_accumulation" skill,
                        #   real updates times of optimizer is train_step/gradient_accumulation_steps
    training_logger = {"train":[], "eval":[]}
    epoch_best_loss = 1e10
    epoch_best_acc = 0
    
    # eval_epoch_best_loss = 1e10
    eval_epoch_best_acc = 0
    BEST_ACC_FLAG = False

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            resume_step = None
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        if "epoch" in path:
            args.num_train_epochs -= int(path.replace("epoch_", ""))
        else:
            resume_step = int(path.replace("step_", ""))
            args.num_train_epochs -= resume_step // len(train_dataloader)
            resume_step = (args.num_train_epochs * len(train_dataloader)) - resume_step

    for epoch in range(args.num_train_epochs):
        model.train()
        # if args.with_tracking:
        #     total_loss = 0
        
        # train metric
        total_loss = 0
        cum_avg_batch_loss_List = []
        epoch_loss = 0
        
        total_acc = 0
        cum_avg_batch_acc_List = []
        epoch_acc = 0
        # batch_best_acc = 0
        
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        for train_step, (batch, batch_examples, batch_dataset) in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == 0 and train_step < resume_step:
                continue
            
            all_start_logits = []
            all_end_logits = []
            batch_acc = 0
            cum_avg_batch_loss = 0 # cum_avg_batch_loss = total_loss / (train_step + 1)
            cum_avg_batch_acc = 0 # cum_avg_batch_acc = total_acc / (train_step + 1)
            
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            # Print outputs
            # print("="*100, "\n", "=> outputs\n", type(outputs), outputs, "\n")
            # print("=> start_logits\n", type(start_logits), f"len(start_logits) = {len(start_logits)}", "\n") # len(start_logits) = batch_size
            # print("=> end_logits\n", type(end_logits), f"len(end_logits) = {len(end_logits)}", "\n") # len(end_logits) = batch_size
            # input("=> Section: model.train(), print outputs, press Any key to continue")
            
            loss = outputs.loss
            command_info = "="*100 + "\n" + "=> trian\n" + f"train_step = {train_step}, completed_steps = {completed_steps}\n"
            tqdm.write(command_info)
            command_info = f"curr_batch_loss {type(loss)} {loss}\n"
            tqdm.write(command_info)
            # We keep track of the loss at each epoch
            # if args.with_tracking:
            #     total_loss += loss.detach().float()
            total_loss += loss.detach().float()
            cum_avg_batch_loss = total_loss.cpu().detach().numpy() / (train_step + 1)
            cum_avg_batch_loss_List.append(cum_avg_batch_loss)
            
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if train_step % args.gradient_accumulation_steps == (args.gradient_accumulation_steps-1) or train_step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
            
            all_start_logits.append(accelerator.gather(start_logits).cpu().detach().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().detach().numpy())

            max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

            # concatenate the numpy array
            start_logits_concat = create_and_fill_np_array(all_start_logits, batch_dataset, max_len)
            end_logits_concat = create_and_fill_np_array(all_end_logits, batch_dataset, max_len)

            # delete the list of numpy arrays
            del all_start_logits
            del all_end_logits
            outputs_numpy = (start_logits_concat, end_logits_concat)
            # print("="*100, "\n", "=> outputs_numpy\n", type(outputs_numpy), outputs_numpy, f", len = {len(outputs_numpy)}", "\n")
            # print("="*100, "\n", "=> batch_examples\n", type(batch_examples), batch_examples, f", len = {len(batch_examples)}", "\n")
            # print("="*100, "\n", "=> batch_dataset\n", type(batch_dataset), batch_dataset, f", len = {len(batch_dataset)}", "\n")
            # input("=> Section: model.train() -> print outputs_numpy, press Any key to continue")
            prediction = post_processing_function(batch_examples, batch_dataset, outputs_numpy)
            
            # Calculate average accuracy
            train_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
            total_acc += train_metric["exact_match"]
            cum_avg_batch_acc = total_acc / (train_step + 1)
            cum_avg_batch_acc_List.append(cum_avg_batch_acc)
            tqdm.write(f"current batch_acc = {batch_acc}, cum_avg_batch_acc = {cum_avg_batch_acc}, cum_avg_batch_loss = {cum_avg_batch_loss}\n")
            
            if isinstance(checkpointing_steps, int): # isinstance() 函數來判斷一個對像是否是一個已知的類型，類似 type()。
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        # end of model.train()
        epoch_loss = cum_avg_batch_loss # here, cum_avg_batch_loss = total average loss for one batch
        if epoch_loss < epoch_best_loss: epoch_best_loss = epoch_loss
        epoch_acc = cum_avg_batch_acc
        if epoch_acc > epoch_best_acc: epoch_best_acc = epoch_acc
        
        print("="*100, "\n", f"train_step = {train_step}, completed_steps = {completed_steps}\n")
        # Update training_logger
        train_log = { "epoch": epoch,
                      "cum_avg_batch_loss": cum_avg_batch_loss_List, # Tensor
                      "epoch_loss": epoch_loss, # Tensor
                      "epoch_best_loss": epoch_best_loss, # Tensor
                      "cum_avg_batch_acc": cum_avg_batch_acc_List,
                      "epoch_acc": epoch_acc,
                    #   "batch_best_acc": batch_best_acc,
                      "epoch_best_acc": epoch_best_acc,
                      "total_completed_steps (optimizer update)": completed_steps
                    }
        training_logger["train"].append(train_log)
        print(f"training_logs['train'] = \n{training_logger['train']}\n")
        # input("=> Section: model.train() -> print training_logger['train'], press Any key to continue")
        
        # Evaluation
        print("="*100, "\n")
        logger.info("***** Running Evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
        
        # eval metric
        # total_loss = 0
        # cum_avg_batch_loss_List = []
        # epoch_loss = 0
        epoch_acc = 0
        
        all_start_logits = []
        all_end_logits = []

        progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        for eval_step, batch in enumerate(eval_dataloader):
            
            cum_avg_batch_loss = 0
            
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                # print("="*100, "\n", "=> outputs ", type(outputs), "\n")
                # print(outputs, "\n")
                # print("=> outputs.start_logits ", type(start_logits), "\n")
                # print(start_logits, "\n")
                # print("=> outputs.end_logits ", type(end_logits), "\n")
                # print(end_logits, "\n")
                # input("=> Section: model.eval(), print(outputs, outputs.start_logits, outputs.end_logits), press Any key to continue")

                # # Calculate loss
                # loss = outputs.loss
                # print("="*100, "\n", "=> eval_loss", type(loss), "\n")
                # print(len(loss["start_logits"]), len(loss["start_logits"]), "\n") => 8, 8
                # print(loss, "\n")
                # total_loss += loss.detach().float()
                # cum_avg_batch_loss = total_loss / (train_step + 1)
                # cum_avg_batch_loss_List.append(cum_avg_batch_loss)
                progress_bar.update(1)

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len) # len(start_logits_concat) = len(eval_dataloader)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len) # len(end_logits_concat) = len(eval_dataloader)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits
        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
        # end of Evaluation loop and predition postprocessing
        
        # epoch_loss = cum_avg_batch_loss # here, cum_avg_batch_loss = total average loss for one batch
        # if epoch_loss < eval_epoch_best_loss: eval_epoch_best_loss = epoch_loss

        # Calculate average accuracy
        eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        epoch_acc = eval_metric["exact_match"]
        if epoch_acc > eval_epoch_best_acc: # store best accuracy
            BEST_ACC_FLAG = True
            print("="*100, "\n", "Best "*10)
            eval_epoch_best_acc = epoch_acc
        
        # Update training_logger
        eval_log = { "epoch": epoch,
                    #  "cum_avg_batch_loss": cum_avg_batch_loss_List,
                     "epoch_acc": epoch_acc,
                     "epoch_best_acc": eval_epoch_best_acc,
                    #  "epoch_loss": epoch_loss,
                    #  "epoch_best_loss": eval_epoch_best_loss,
                   }
        # accelerator.log(log)
        training_logger["eval"].append(eval_log)
        print("="*100, "\n", f"training_logs['eval'] = \n{training_logger['eval']}\n")
        # input("=> Section: model.eval() -> print training_logger['eval'], press Any key to continue")
        
        # Save training_logs
        with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
            json.dump(training_logger, f, indent=2) # indent => 縮排，沒加 indent 所有資料在 file 內會變成一行
            print(f"training logs saved in {args.output_dir}/training_logs.json\n")
            # NOTE: training logs saved in ./tmp/QA_SaveDir/[logger]Training_log.json


        # # Save(@best) Configuration, Model_weights, tokenizer_config, Special_tokens
        if args.output_dir is not None and BEST_ACC_FLAG:
            BEST_ACC_FLAG = False
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                # if args.push_to_hub:
                #     repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
        
        # Save accelerator_state
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir) # output_dir = args.output_dir/epoch_{epoch}
            # print("accelerator save ! ", "="*50)
            accelerator.save_state(output_dir)
    # end of train
    
    answer_out = prediction.predictions
    # print("="*100, "\n", "=> answer_out\n", type(answer_out), f", len = {len(answer_out)}\n", answer_out, "\n")
    write_csv(args.prediction_csv_dir, answer_out)
  
    # # Prediction ( 因為跟 Evaluation的操作一模一樣所以先註解掉 )
    # if args.do_predict:
    #     logger.info("***** Running Prediction *****")
    #     logger.info(f"  Num examples = {len(predict_dataset)}")
    #     logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    #     all_start_logits = []
    #     all_end_logits = []
    #     for step, batch in enumerate(predict_dataloader):
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #             start_logits = outputs.start_logits
    #             end_logits = outputs.end_logits

    #             if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
    #                 start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
    #                 end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

    #             all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
    #             all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    #     max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        
    #     # concatenate the numpy array
    #     start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
    #     end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

    #     # delete the list of numpy arrays
    #     del all_start_logits
    #     del all_end_logits

    #     outputs_numpy = (start_logits_concat, end_logits_concat)
    #     prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
    #     predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    #     logger.info(f"Predict metrics: {predict_metric}")
    #
    # end of Prediction
    
    
    # if args.with_tracking:
    #     log = {
    #         "squad_v2" if args.version_2_with_negative else "squad": eval_metric,
    #         "train_loss": total_loss,
    #         "epoch": epoch,
    #         "step": completed_steps,
    #     }
    
    # if args.do_predict:
    #     log["squad_v2_predict" if args.version_2_with_negative else "squad_predict"] = predict_metric


if __name__ == "__main__":
    main()
