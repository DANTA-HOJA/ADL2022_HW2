# ADL2022_HW2
110.2 CSIE5431_深度學習之應用（ADL）

# Environment settings

follow [README.md](https://github.com/huggingface/transformers/tree/main/examples) to install packages using by huggingface/transformers

# Task 1：Context Selection

Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice

use ```run_swag_no_trainer.py```

# Task 2：Question Answering
Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

use ```run_qa_no_trainer.py```


# arg of Multiple Choice

Namespace(

    dataset_name='swag', -> "The name of the dataset to use (via the datasets library).", # will download from "HuggingFace datasets Hub"
    dataset_config_name=None, -> "The configuration name of the dataset to use (via the datasets library).", # from "HuggingFace datasets Hub"
    train_file=None, -> "A csv or a json file containing the training data." # from PC
    validation_file=None, -> "A csv or a json file containing the validation data." # from PC


    model_name_or_path='bert-base-cased', -> "Path to pretrained model or model identifier from huggingface.co/models."
    config_name=None, -> "Pretrained config name or path if not the same as model_name"
    model_type=None, -> "Model type to use if training from scratch."
    

    tokenizer_name=None, -> "Pretrained tokenizer name or path if not the same as model_name"
    use_slow_tokenizer=False, -> "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."


    max_length=384, -> "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                       " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    pad_to_max_length=False, -> "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."
    

    per_device_train_batch_size=1, -> "Batch size (per device) for the training dataloader."
    per_device_eval_batch_size=8, -> "Batch size (per device) for the evaluation dataloader."


    weight_decay=0.0, -> "Weight decay to use."
    gradient_accumulation_steps=2, -> "Number of updates steps to accumulate before performing a backward/update pass."

    
    num_train_epochs=1, -> "Total number of training epochs to perform."
    max_train_steps=None, -> "Total number of training steps to perform. If provided, overrides num_train_epochs."
    
    
    lr_scheduler_type=<SchedulerType.LINEAR: 'linear'>, -> "The scheduler type to use."
    num_warmup_steps=0, -> "Number of steps for the warmup in the lr scheduler."
    learning_rate=3e-05, -> "Initial learning rate (after the potential warmup period) to use."


    output_dir='/tmp/swag/', -> "Where to store the final model."
    seed=None, -> "A seed for reproducible training."


    push_to_hub=False, -> "Whether or not to push the model to the Hub."
    hub_model_id=None, -> "The name of the repository to keep in sync with the local `output_dir`."
    hub_token=None, -> "The token to use to push to the Model Hub."


    debug=False, -> "Activate debug mode and run training only with a subset of data."
    checkpointing_steps=None, -> "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
    resume_from_checkpoint=None, -> "If the training should continue from a checkpoint folder."
    with_tracking=False, -> "Whether to load in all available experiment trackers from the environment and use them for logging."
    
)