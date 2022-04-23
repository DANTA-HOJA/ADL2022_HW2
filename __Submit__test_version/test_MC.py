#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import json
import pandas as pd
import numpy as np
import logging
import math
import os
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
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
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, get_full_repo_name


logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    
    # dataset 
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).", # will download from "HuggingFace datasets Hub"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).", # from "HuggingFace datasets Hub"
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data." # from PC
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data." # from PC
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data." # from PC
    )
    parser.add_argument(
        "--context_file", type=str, default=None,
        help=(
            "A csv or a json file containing the context data." # from PC
            "used to map paragrph_numbers to content_string"
        )
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
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    # sentence
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
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
        # 預熱期間，學習率從0線性（也可非線性）增加到優化器中的初始預設lr，之後使其學習率從優化器中的初始lr線性降低到0
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    
    # training mode and log
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--debug_max_sample",
        type=int,
        default=100,
        help="A number use to slice the dataset during debug_mode, default is 100",
    )
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
    
    # save path
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--predict_file", type=str, default=None, help="Where to save the prediction file.")
    
    # push to hub
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
                
        # label_name = "label" if "label" in features[0].keys() else "labels"
        
        # labels = [feature.pop(label_name) for feature in features] # 先暫時把 labels 拿出來
        id = [feature.pop("id") for feature in features]
        num_second_sentences = [feature.pop("num_second_sentences") for feature in features]
        question = [feature.pop("question") for feature in features]
        # answer_text = [feature.pop("answer_text") for feature in features]
        # answer_start = [feature.pop("answer_start") for feature in features]
        
        # 原始的 sample code 處理
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels（把 labels 加回去）
        # batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        
        return batch, id, num_second_sentences, question #, answer_text, answer_start


def dataset_change_format(data_path:str) -> pd.DataFrame:
    
    # print("="*100, "\n", f"=> Apply dataset_change_format() on '{data_path}'\n")
    
    data_Dict = json.loads(Path(data_path).read_text())
    
    # print(f"Before change: len = {len(data_Dict)}, content_format = \n{data_Dict[0]}\n")
    data_List = list()
    for i in range(len(data_Dict)):
        data_List.append({"id": data_Dict[i]["id"],
                            "question":data_Dict[i]["question"],
                            "context_0":data_Dict[i]["paragraphs"][0],
                            "context_1":data_Dict[i]["paragraphs"][1],
                            "context_2":data_Dict[i]["paragraphs"][2],
                            "context_3":data_Dict[i]["paragraphs"][3],
                            # "relevant":data_Dict[i]["relevant"],
                            # "answer_text":data_Dict[i]["answer"]["text"],
                            # "answer_start":data_Dict[i]["answer"]["start"],
                            })
    # print(f"After change: len = {len(data_List)}, content_format = \n{data_List[0]}\n")
    return pd.DataFrame(data_List)


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    torch.cuda.set_device(0)
    print("=> torch.cuda.set_device(0)")
    # input("=> In accelerator set device 1, press Any key to continue")
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
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name) # github_sample_command -> load_dataset(swag, None)
    else:
        # data_files = {}
        if args.context_file is not None:
            context_mapping = json.loads(Path(args.context_file).read_text())
        if args.train_file is not None:
            df = dataset_change_format(args.train_file)
            raw_datasets_train = Dataset.from_pandas(df)
            # data_files["train"] = args.train_file
        if args.validation_file is not None:
            df = dataset_change_format(args.validation_file)
            raw_datasets_valid = Dataset.from_pandas(df)
            # data_files["validation"] = args.validation_file
        if args.test_file is not None:
            df = dataset_change_format(args.test_file)
            raw_datasets_test = Dataset.from_pandas(df)
        raw_datasets = datasets.DatasetDict({"test": raw_datasets_test})
        # extension = args.train_file.split(".")[-1]
        # raw_datasets = load_dataset(extension, data_files=data_files)
    

    # print("="*100, "\n", f"context_mapping[0]= {context_mapping[0]}", "\n")
    # print(type(raw_datasets), type(raw_datasets['test']), type(raw_datasets['test'][0]), "\n")
    # print(f"raw_datasets.column_names = {raw_datasets.column_names}\n")
    # print(f"raw_datasets['test'][0] = {raw_datasets['test'][0]}\n")
    # input("=> Load test_file, press Any key to continue")
    
        
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(args.debug_max_sample))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # if raw_datasets["test"] is not None:
    #     column_names = raw_datasets["test"].column_names
    # else:
    #     column_names = raw_datasets["validation"].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.    
    question_name = "question"
    context_names = [f"context_{i}" for i in range(4)]
    
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        first_sentences = [[question] * 4 for question in examples[question_name]]
        num_second_sentences = list()
        for i in range(len(examples["id"])): num_second_sentences.append([examples[context][i] for context in context_names])
        str_second_sentences = list()
        for i in range(len(examples["id"])): str_second_sentences.append([context_mapping[examples[context][i]] for context in context_names])
        # labels = list()
        # for i in range(len(num_second_sentences)): # len(num_second_sentences) == 1000
        #     for sentence_idx in range(len(num_second_sentences[i])): # len(num_second_sentences[i]) == 4
        #         if examples[relevant_name][i] == num_second_sentences[i][sentence_idx]:
        #             labels.append(sentence_idx)
        
        assert len(first_sentences)==len(num_second_sentences)==len(str_second_sentences), "Error: total length not match "
        
        # print("="*100, "\n", f"first_sentences = {first_sentences[0]}\n")
        # print(f"num_second_sentences = {num_second_sentences[0]}\n")
        # print(f"str_second_sentences = {str_second_sentences[0]}\n")
        # print(f"labels = {labels[0]}\n")
        # input("=> In preprocess_function, press Any key to continue")

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        str_second_sentences = list(chain(*str_second_sentences))
        
        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            str_second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs["labels"] = labels
        tokenized_inputs["id"] = examples["id"]
        tokenized_inputs["num_second_sentences"] = num_second_sentences
        tokenized_inputs["question"] = examples["question"]
        # tokenized_inputs["answer_text"] = examples["answer_text"]
        # tokenized_inputs["answer_start"] = examples["answer_start"]
        
        # print("="*100, "\n", type(tokenized_inputs), "\n")
        # df_token_in = pd.DataFrame(tokenized_inputs)
        # print(df_token_in, "\n")
        # print(df_token_in.columns, "\n")
        # print(df_token_in["num_second_sentences"], "\n")
        # input("=> In preprocess_function, \"tokenized_inputs\", press Any key to continue")
        
        return tokenized_inputs

    # dataset preprocess
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["test"].column_names
        )
    # train_dataset = processed_datasets["test"]
    test_dataset = processed_datasets["test"]
    # Print "train_dataset"
    # print("="*100, "\n", "=> test:\n")
    # print(type(test_dataset), "\n")
    # print(test_dataset.column_names, "\n")
    # print(test_dataset[0], "\n\n")
    # input("=> test_dataset preprocess complete, press Any key to continue")


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
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    progress_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)
    QA_sheet = list()
    
    model.eval()
    for test_step, (batch, id, num_second_sentences, question) in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        # print("="*100, "\n", f"=> _step = {test_step}\n")
        # print(id, num_second_sentences, question, f"batch = \n{batch}\n")
        # print(f"outputs = {outputs}\n")
        # print(f"predictions = {predictions}\n")
        
        # 將預測根據 label 回推文本並蒐集成 .csv 以供 QA 使用
        for i in range(len(predictions)):
            pred_label = predictions[i].detach().cpu().numpy().tolist()
            # print(id[i], context_mapping[num_second_sentences[i][pred_label]], question[i])
            QA_sheet.append({"id": id[i],
                                "context": context_mapping[num_second_sentences[i][pred_label]],
                                "question": question[i],
                                # "answers":{
                                #     "answer_start": [answer_start[i],],
                                #     "text": [answer_text[i],],
                                # }
                            })
        # print(QA_sheet[0])
        df_QA_sheet = pd.DataFrame(QA_sheet)
        # print(f"df_QA_sheet.shape = {df_QA_sheet.shape}\n")
        progress_bar.update(1)
    # end of model.eval()
    
    df_QA_sheet.to_json(args.predict_file, indent=2, force_ascii=False, orient="records")
    print(f"Save predict output, path = {args.predict_file}\n")
    # input("=> Section: end of model.eval(), save QA_sheet, press Any key to continue")

# end of test

if __name__ == "__main__":
    main()
