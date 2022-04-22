parser(
    
    # dataset
    "--dataset_name", -> type=str, default=None, help="The name of the dataset to use (via the datasets library).",
    "--dataset_config_name", -> type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).",
    "--train_file", -> type=str, default=None, help="A csv or a json file containing the training data.",
    "--validation_file", -> type=str, default=None, help="A csv or a json file containing the validation data.",
    "--test_file", -> type=str, default=None, help="A csv or a json file containing the Prediction data."
    "--context_file", type=str, default=None, help="A csv or a json file containing the context data. used to map paragrph_numbers to content_string"


    # sample limit
    "--max_train_samples", -> type=int, default=None, help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    "--max_eval_samples", -> type=int, default=None, help="For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.",
    "--max_predict_samples", -> type=int, default=None, help="For debugging purposes or quicker training, truncate the number of prediction examples to this",


    # model
    "--model_name_or_path", -> type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True,
    "--config_name",-> type=str, default=None, help="Pretrained config name or path if not the same as model_name", # for HW2-Q4 -> "bert"
    "--model_type", -> type=str, default=None, help="Model type to use if training from scratch.", choices=MODEL_TYPES,


    # tokenizer
    "--tokenizer_name", -> type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name",
    # "--use_slow_tokenizer", -> action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    
    
    # sentence preprocessing
    "--max_seq_length", -> type=int, default=384, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                                                       " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    "--pad_to_max_length", -> action="store_true", help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    "--doc_stride", -> type=int, default=128, help="When splitting up a long document into chunks how much stride to take between chunks.",
    "--preprocessing_num_workers", -> type=int, default=4, help="How many worker can do preprocessing."
    "--overwrite_cache", -> type=bool, default=False, help="Overwrite the cached training and evaluation sets"

    # batch_size
    "--per_device_train_batch_size", -> type=int, default=8, help="Batch size (per device) for the training dataloader.",
    "--per_device_eval_batch_size", -> type=int, default=8, help="Batch size (per device) for the evaluation dataloader.",


    # weight and gradient
    "--weight_decay", -> type=float, default=0.0, help="Weight decay to use."
    "--gradient_accumulation_steps", -> type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",


    # train_loop
    "--num_train_epochs", -> type=int, default=3, help="Total number of training epochs to perform."
    "--max_train_steps", -> type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.",


    # learning_rate
    "--lr_scheduler_type", -> type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    "--learning_rate", -> type=float, default=5e-5, help="Initial learning rate (after the potential warmup period) to use.",


    # training mode and log
    "--checkpointing_steps", -> type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    "--resume_from_checkpoint", -> type=str, default=None, help="If the training should continue from a checkpoint folder.",
    "--with_tracking", -> action="store_true", help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    "--seed", type=int, default=None, help="A seed for reproducible training."


    # predition
    "--do_predict", -> action="store_true", help="To do prediction on the question answering model"
    "--max_answer_length", -> type=int, default=30, help="The maximum length of an answer that can be generated. This is needed because the start "
                                                         "and end predictions are not conditioned on one another.",
    "--n_best_size", -> type=int, default=20, help="The total number of n-best predictions to generate when looking for an answer.",


    # squad version2 -> if your dataset contains samples with no possible answers (like SQuAD version 2), you need to pass along this flag
    "--version_2_with_negative", -> type=bool, default=False, help="If true, some of the examples do not have an answer.",
    "--null_score_diff_threshold", -> type=float, default=0.0, help="The threshold used to select the null answer: if the best answer has a score that is less than "
                                                                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                                                                    "Only useful when `version_2_with_negative=True`.",


    # save path
    "--output_dir", -> type=str, default=None, help="Where to store the final model."
    "--prediction_csv_dir", -> type=str, default=None, help="Where to store the predict result in a csv file."

    # push to hub
    "--push_to_hub", -> action="store_true", help="Whether or not to push the model to the Hub."
    "--hub_model_id", -> type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    "--hub_token", type=str, help="The token to use to push to the Model Hub."
    
)


# Train dataset extraction and Preprocessing
if "train" not in raw_datasets:
    raise ValueError("--do_train requires a train dataset") # --do_train 應該是 with_trainer 的 args，雖然 no_trainer 版本內沒有這個 args 但描述就順用
train_dataset = raw_datasets["train"]
if args.max_train_samples is not None:
    # We will select sample from whole data if agument is specified
    train_dataset = train_dataset.select(range(args.max_train_samples))
# Create train feature from dataset
with accelerator.main_process_first():
    train_dataset = train_dataset.map(
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

train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

===> train_dataset = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']


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

eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
eval_dataloader = DataLoader(
    eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
)

===> eval_examples = ['id', 'title', 'context', 'question', 'answers']
===> eval_dataset = ['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'example_id'] # example_id is original "id"


# Command separator_bar format
"="*100, "\n", 