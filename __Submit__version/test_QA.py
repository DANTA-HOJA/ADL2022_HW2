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
# check_min_version("4.19.0.dev0")

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
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
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
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the testing dataloader.",
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


def dataset_change_format(data_path:str, context_path:str) -> pd.DataFrame:
    
    # print("="*100, "\n", f"=> Apply dataset_change_format() on '{data_path}'\n")
    
    context_mapping = json.loads(Path(context_path).read_text())
    data_Dict = json.loads(Path(data_path).read_text())
    
    # print(f"Before change: len = {len(data_Dict)}, content_format = \n{data_Dict[0]}\n")
    data_List = list()
    for i in range(len(data_Dict)):
        data_List.append({"id": data_Dict[i]["id"],
                          "context":data_Dict[i]["context"],
                          "question":data_Dict[i]["question"],
                        #   "answers":{
                        #       "text":[ data_Dict[i]["answer"]["text"], ],
                        #       "answer_start":[ data_Dict[i]["answer"]["start"], ],
                        #   }
                        })
    # print(f"After change: len = {len(data_List)}, content_format = \n{data_List[0]}\n")
    return pd.DataFrame(data_List)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # torch.cuda.set_device(1)
    # print("=> torch.cuda.set_device(1)")
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
        if args.test_file is not None:
            df = dataset_change_format(args.test_file, args.context_file)
            raw_datasets_test = Dataset.from_pandas(df)
            # data_files["validation"] = args.validation_file
        # if args.test_file is not None:
        #     # data_files["test"] = args.test_file
        raw_datasets = datasets.DatasetDict({"test": raw_datasets_test})
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
    
    # print("="*100, "\n", type(raw_datasets), type(raw_datasets['test']), type(raw_datasets['test'][0]), "\n")
    # print(f"raw_datasets.column_names = {raw_datasets.column_names}\n")
    # print(f"raw_datasets['test'][0] = {raw_datasets['test'][0]}\n")
    # input("=> Load test_file, press Any key to continue")
    

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

    column_names = raw_datasets["test"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    # answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length) # 最大 seq_length只能是 tokenizer max 因為超過就不在 tokenizer 的 train domain 了


    # Test dataset preprocessing function
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
    
    # Test dataset extraction and Preprocessing
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset") # --do_eval 應該是 with_trainer 的 args，雖然 no_trainer 版本內沒有這個 args 但描述就順用
    test_examples = raw_datasets["test"]
    if args.max_predict_samples is not None:
        # We will select sample from whole data
        test_examples = test_examples.select(range(args.max_predict_samples))
    # Testing Feature Creation
    with accelerator.main_process_first():
        test_dataset = test_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            test_dataset = test_dataset.select(range(args.max_predict_samples))

    # print("="*100, "\n", type(test_examples), "\n") # eval_examples = 未前處理 # <class 'datasets.arrow_dataset.Dataset'>
    # print(test_examples.column_names, "\n") # ['id', 'context', 'question', 'answers']
    # print(test_examples[0], "\n\n")
    # print(type(test_dataset), "\n") # eval_dataset = 已前處理 # <class 'datasets.arrow_dataset.Dataset'>
    # print(test_dataset.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id']
    # print(test_dataset[0], "\n")
    # input("=> Test dataset extraction and Preprocessing complete, press Any key to continue")


    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        

    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataloader = DataLoader(
        test_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_test_batch_size
    )
    # print("="*100, "\n", type(test_dataset_for_model), "\n")
    # print(test_dataset_for_model.column_names, "\n") # ['input_ids', 'token_type_ids', 'attention_mask']
    # input("=> Test DataLoader ready, press Any key to continue")

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

        # references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        # print("="*100, "\n", references, "\n")
        # input("-> In post_processing_function(), print(references), press Any key to continue")
        
        return formatted_predictions


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


    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )


    all_start_logits = []
    all_end_logits = []

    progress_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)
    for test_step, batch in enumerate(test_dataloader):
        
        # cum_avg_batch_loss = 0
        
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            progress_bar.update(1)

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len) # len(start_logits_concat) = len(eval_dataloader)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len) # len(end_logits_concat) = len(eval_dataloader)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits
    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(test_examples, test_dataset, outputs_numpy)
    # end of test loop and postprocessing
    
    answer_out = prediction
    # print("="*100, "\n", "=> answer_out\n", type(answer_out), f", len = {len(answer_out)}\n", answer_out, "\n")
    # input("=> before write predict file, press Any key to continue")
    write_csv(args.prediction_csv_dir, answer_out)




if __name__ == "__main__":
    main()
