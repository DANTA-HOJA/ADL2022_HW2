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
from datasets import load_dataset, load_metric
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
        help="A number use to slice the dataset during debug_mode.",
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
    
    # save and reproducible
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
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
                
        label_name = "label" if "label" in features[0].keys() else "labels"
        
        labels = [feature.pop(label_name) for feature in features] # 先暫時把 labels 拿出來
        id = [feature.pop("id") for feature in features]
        num_second_sentences = [feature.pop("num_second_sentences") for feature in features]
        question = [feature.pop("question") for feature in features]
        answer_text = [feature.pop("answer_text") for feature in features]
        answer_start = [feature.pop("answer_start") for feature in features]
        
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
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        
        return batch, id, num_second_sentences, question, answer_text, answer_start

def Dataset_de_Hierarchical(data_path:str) -> pd.DataFrame:
        data_Path = Path(data_path)
        data_Dict = json.loads(data_Path.read_text())
        print(f"Before de_Hierarchical: len = {len(data_Dict)}, content_format = \n{data_Dict[0]}\n")
        data_List = list()
        for i in range(len(data_Dict)):
            data_List.append({"id": data_Dict[i]["id"],
                            "question":data_Dict[i]["question"],
                            "context_0":data_Dict[i]["paragraphs"][0],
                            "context_1":data_Dict[i]["paragraphs"][1],
                            "context_2":data_Dict[i]["paragraphs"][2],
                            "context_3":data_Dict[i]["paragraphs"][3],
                            "relevant":data_Dict[i]["relevant"],
                            "answer_text":data_Dict[i]["answer"]["text"],
                            "answer_start":data_Dict[i]["answer"]["start"],
                            })
        print(f"After de_Hierarchical: len = {len(data_List)}, content_format = \n{data_List[0]}\n")
        return pd.DataFrame(data_List)

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
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
        data_files = {}
        if args.train_file is not None:
            Dataset_de_Hierarchical(args.train_file).to_csv("./train_MC.csv", index=False, encoding="utf_8_sig")
            data_files["train"] = "./train_MC.csv"
        if args.validation_file is not None:
            Dataset_de_Hierarchical(args.validation_file).to_csv("./valid_MC.csv", index=False, encoding="utf_8_sig")
            data_files["validation"] = "./valid_MC.csv"
        # extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset("csv", data_files=data_files)
        
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            assert args.debug_max_sample is not None, "args.debug_max_sample is required"
            raw_datasets[split] = raw_datasets[split].select(range(args.debug_max_sample))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    context_mapping = json.loads(Path(args.context_file).read_text())
    print(f"context_mapping[0]= {context_mapping[0]}")
    
    question_name = "question"
    context_names = [f"context_{i}" for i in range(4)]
    relevant_name = "relevant"
    print(f"question_name = {question_name}")
    print(f"context_name = {context_names}")
    print(f"relevant_name = {relevant_name}")
    
    print(f"raw_datasets['train'] -> len = {len(raw_datasets['train'])}, type() = {type(raw_datasets['train'])}, content_format = \n{raw_datasets['train'][0]}\n")

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
        labels = list()
        for i in range(len(num_second_sentences)): # len(num_second_sentences) == 1000
            for sentence_idx in range(len(num_second_sentences[i])): # len(num_second_sentences[i]) == 4
                if examples[relevant_name][i] == num_second_sentences[i][sentence_idx]:
                    labels.append(sentence_idx)
        
        assert len(first_sentences)==len(num_second_sentences)==len(str_second_sentences), "Error: total length not match "
        
        print(f"first_sentences = {first_sentences[0]}")
        print(f"num_second_sentences = {num_second_sentences[0]}")
        print(f"str_second_sentences = {str_second_sentences[0]}")
        print(f"labels = {labels[0]}")
        # input("in preprocess_function => press Any key to continue")

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
        tokenized_inputs["labels"] = labels
        tokenized_inputs["id"] = examples["id"]
        tokenized_inputs["num_second_sentences"] = num_second_sentences
        tokenized_inputs["question"] = examples["question"]
        tokenized_inputs["answer_text"] = examples["answer_text"]
        tokenized_inputs["answer_start"] = examples["answer_start"]
        
        print(type(tokenized_inputs))
        df_token_in = pd.DataFrame(tokenized_inputs)
        print(df_token_in)
        print(df_token_in.columns)
        print(df_token_in["num_second_sentences"])
        # input("in preprocess_function => press Any key to continue")
        
        return tokenized_inputs

    # dataset preprocess
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
        )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    # input("Preprocess complete => press Any key to continue")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # input("Dataset & DataLoader setup complete => press Any key to continue")

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
    print(f"model.named_parameters = \n{model.named_parameters}")
    # input(f"Optimizer and Move to accelerator_device {device} => press Any key to continue")

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
        num_training_steps=args.max_train_steps, # total time of updates for optimizer 
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
    #     accelerator.init_trackers("swag_no_trainer", experiment_config)

    # Metrics
    metric = load_metric("accuracy")

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
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Train counter, logs, parameters
    completed_steps = 0
    training_logger = list()
    best_avg_loss = 1e10
    curr_avg_loss = 0
    best_acc = 0
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
        total_loss = 0
        for train_step, (batch, id, num_second_sentences, question, answer_text, answer_start) in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == 0 and train_step < resume_step:
                continue
            outputs = model(**batch) # (**name) -> 自動將一個 batch 展開送給 model````
            print(f"train_step = {train_step}, completed_steps = {completed_steps}, total_loss = {total_loss}")
            print(id, num_second_sentences, question, answer_text, answer_start, f"batch = \n{batch}")
            print(f"outputs = {outputs}")
            # input("Section: model.train() -> print outputs, press Any key to continue ")
            loss = outputs.loss
            # We keep track of the loss at each epoch
            # if args.with_tracking:
            #     total_loss += loss.detach().float()
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            # 當 gradient_accumulation_steps=2 時，由於 train_step 從 0 開始所以當 train_step=1 loss 已經累積 2 次，因此餘數為 gradient_accumulation_steps-1 應更新一次 optimizer
            if train_step % args.gradient_accumulation_steps == (args.gradient_accumulation_steps-1) or train_step == len(train_dataloader) - 1: # Gradient Accumulation already be implemented
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int): # isinstance() 函數來判斷一個對像是否是一個已知的類型，類似 type()。
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        # end of model.train()
        
        # out file
        QA_sheet = list()
        
        model.eval()
        for eval_step, (batch, id, num_second_sentences, question, answer_text, answer_start) in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            print(f"eval_step = {eval_step}")
            print(id, num_second_sentences, question, answer_text, answer_start, f"batch = \n{batch}")
            print(f"outputs = {outputs}")
            print(f"predictions = {predictions}")
            
            # 將預測根據 label 回推文本並蒐集成 .csv 以供 QA 使用
            for i in range(len(batch["labels"])):
                pred_label = predictions[i].detach().cpu().numpy().tolist()
                print(id[i], context_mapping[num_second_sentences[i][pred_label]], question[i], answer_text[i], answer_start[i])
                QA_sheet.append({"id": id[i],
                                 "context": context_mapping[num_second_sentences[i][pred_label]],
                                 "question": question[i],
                                 "answer_text": answer_text[i],
                                 "answer_start": answer_start[i]
                                })
            df_QA_sheet = pd.DataFrame(QA_sheet)
            df_QA_sheet.to_csv("./QA_sheet.csv", index=False, encoding="utf_8_sig")
            # input("Section: model.eval() -> save QA_sheet, press Any key to continue ")
            
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        # end of model.eval()
        
        # Calculate average accuracy
        eval_metric = metric.compute() # eval_accuracy = {'accuracy': 0.86}
        if eval_metric["accuracy"] > best_acc: # store best accuracy
            BEST_ACC_FLAG = True
            print("Best "*10)
            best_acc = eval_metric["accuracy"]
            
        # Calculate average loss
        curr_avg_loss = (total_loss/(completed_steps*args.gradient_accumulation_steps)).detach().cpu().numpy().tolist()
        if curr_avg_loss < best_avg_loss: # store best loss
            best_avg_loss = curr_avg_loss

        # Update loggers
        print(f"train_step = {train_step}, completed_steps = {completed_steps}")
        log = { "epoch": epoch,
                "eval_accuracy": eval_metric["accuracy"],
                "best_accuracy": best_acc,
                "curr_avg_loss": curr_avg_loss,
                "best_avg_loss": best_avg_loss,
                "total_completed_steps (optimizer update)": completed_steps
              }
        accelerator.log(log)
        training_logger.append(log)
        print(f"training_logs = \n{training_logger}")
        # input("Section: model.eval() -> print training_logger, press Any key to continue ")
        
        # Save training_logs
        with open(os.path.join(args.output_dir, "training_logs.json"), "w") as f:
            json.dump(training_logger, f, indent=2) # indent => 縮排，沒加 indent 所有資料在 file 內會變成一行
            print(f"training logs saved in {args.output_dir}/training_logs.json")
            # NOTE: training logs saved in ./tmp/MC_SaveDir/[logger]Training_log.json
        
        # # Save(@best) Configuration, Model_weights, tokenizer_config, Special_tokens
        # if args.push_to_hub and epoch < args.num_train_epochs - 1:
        if args.output_dir is not None and BEST_ACC_FLAG:
            BEST_ACC_FLAG = False
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # print("model save ! ", "="*50)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            # NOTE: Configuration saved in ./tmp/MC_SaveDir/config.json
            # NOTE: Model weights saved in ./tmp/MC_SaveDir/pytorch_model.bin
            # print("DONE model save ! ", "="*50)
            if accelerator.is_main_process:
                # print("tokenizer save ! ", "="*50)
                tokenizer.save_pretrained(args.output_dir)
                # NOTE: tokenizer config file saved in ./tmp/MC_SaveDir/tokenizer_config.json
                # NOTE: Special tokens file saved in ./tmp/MC_SaveDir/special_tokens_map.json
                # print("DONE tokenizer save ! ", "="*50)
                # repo.push_to_hub(
                #     commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                # )
                
        # Save accelerator_state
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir) # output_dir = args.output_dir/epoch_{epoch}
            # print("accelerator save ! ", "="*50)
            accelerator.save_state(output_dir)
            # NOTE: - INFO - accelerate.accelerator - Saving current state to ./tmp/MC_SaveDir/["epoch_{epoch}"]
            # NOTE: - INFO - accelerate.checkpointing - Model weights saved in ./tmp/MC_SaveDir/["epoch_{epoch}"]/pytorch_model.bin
            # NOTE: - INFO - accelerate.checkpointing - Optimizer state saved in ./tmp/MC_SaveDir/["epoch_{epoch}"]/optimizer.bin
            # NOTE: - INFO - accelerate.checkpointing - Gradient scaler state saved in ./tmp/MC_SaveDir/["epoch_{epoch}"]/scaler.pt
            # NOTE: - INFO - accelerate.checkpointing - Random states saved in ./tmp/MC_SaveDir/["epoch_{epoch}"]/random_states_0.pkl
            # print("DONE accelerator save ! ", "="*50)
    # end of train

if __name__ == "__main__":
    main()
