Multiple_Choice_Namespace( 
                          
    # partial parameter are set using sample_command mentioned in README.md：https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice

    # dataset
    dataset_name='swag', -> "The name of the dataset to use (via the datasets library).", # will download from "HuggingFace datasets Hub"
    dataset_config_name=None, -> "The configuration name of the dataset to use (via the datasets library).", # from "HuggingFace datasets Hub"
    train_file=./Dataset/train.json, -> "A csv or a json file containing the training data." # from PC
    validation_file=./Dataset/valid.json, -> "A csv or a json file containing the validation data." # from PC
    context_file=./Dataset/context.json -> "A csv or a json file containing the context data. used to map paragrph_numbers to content_string" # from PC
    
    # model
    model_name_or_path='bert-base-cased', -> "Path to pretrained model or model identifier from huggingface.co/models."
    config_name=None, -> "Pretrained config name or path if not the same as model_name"
    model_type=None, -> "Model type to use if training from scratch."  # for HW2-Q4 -> "bert"
    
    # tokenizer
    tokenizer_name=None, -> "Pretrained tokenizer name or path if not the same as model_name"
    use_slow_tokenizer=False, -> "If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library)."

    # sentence
    max_length=384, -> "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
                       " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    pad_to_max_length=False, -> "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."
    
    # batch_size
    per_device_train_batch_size=1, -> "Batch size (per device) for the training dataloader."
    per_device_eval_batch_size=8, -> "Batch size (per device) for the evaluation dataloader."

    # weight and gradient
    weight_decay=0.0, -> "Weight decay to use."
    gradient_accumulation_steps=2, -> "Number of updates steps to accumulate before performing a backward/update pass."

    # train_loop
    num_train_epochs=1, -> "Total number of training epochs to perform."
    max_train_steps=None, -> "Total number of training steps to perform. If provided, overrides num_train_epochs." # 每 gradient_accumulation_steps 更新一次 optimizer，更新一次算一個 train_step，
                                                                                                                   # 傳統上美處理一個 batch 等於 "one" train_step 並直接更新 optimizer 一次，
                                                                                                                   # 但為了解決 run out of GPU memory 的問題而導入 gradient accumulation，
                                                                                                                   # 因此 optimizer更新變成 batch_gradient 累計 gradient_accumulation 次才更新一次。
                                                                                                                   
    
    # learning_rate
    lr_scheduler_type=<SchedulerType.LINEAR: 'linear'>, -> "The scheduler type to use."
    num_warmup_steps=0, -> "Number of steps for the warmup in the lr scheduler."
    learning_rate=3e-05, -> "Initial learning rate (after the potential warmup period) to use."

    # training mode and log
    debug=False, -> "Activate debug mode and run training only with a subset of data."
    checkpointing_steps="epoch", -> "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
    resume_from_checkpoint=None, -> "If the training should continue from a checkpoint folder."
    with_tracking=False, -> "Whether to load in all available experiment trackers from the environment and use them for logging."
    
    # save and reproducible
    output_dir='/tmp/swag/', -> "Where to store the final model."
    seed=None, -> "A seed for reproducible training."

    # push to hub
    push_to_hub=False, -> "Whether or not to push the model to the Hub."
    hub_model_id=None, -> "The name of the repository to keep in sync with the local `output_dir`."
    hub_token=None, -> "The token to use to push to the Model Hub."
    
)


processed_datasets = Index(['input_ids', 'token_type_ids', 'attention_mask', 'labels', 'id', 'num_second_sentences', 'question', 'answer_text', 'answer_start'], dtype='object')


Model config BertConfig {
  "_name_or_path": "bert-base-chinese",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.19.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}


eval_metric = metric.compute() # eval_accuracy = {'accuracy': 0.86}


Section: model.eval() -> print training_logger, press Any key to continue
model save !  ==================================================
Configuration saved in ./tmp/MC_SaveDir/config.json
Model weights saved in ./tmp/MC_SaveDir/pytorch_model.bin
DONE model save !  ==================================================
tokenizer save !  ==================================================
tokenizer config file saved in ./tmp/MC_SaveDir/tokenizer_config.json
Special tokens file saved in ./tmp/MC_SaveDir/special_tokens_map.json
DONE tokenizer save !  ==================================================
accelerator save !  ==================================================
04/18/2022 06:22:16 - INFO - accelerate.accelerator - Saving current state to ./tmp/MC_SaveDir/epoch_0
04/18/2022 06:22:20 - INFO - accelerate.checkpointing - Model weights saved in ./tmp/MC_SaveDir/epoch_0/pytorch_model.bin
04/18/2022 06:22:26 - INFO - accelerate.checkpointing - Optimizer state saved in ./tmp/MC_SaveDir/epoch_0/optimizer.bin
04/18/2022 06:22:26 - INFO - accelerate.checkpointing - Gradient scaler state saved in ./tmp/MC_SaveDir/epoch_0/scaler.pt
04/18/2022 06:22:26 - INFO - accelerate.checkpointing - Random states saved in ./tmp/MC_SaveDir/epoch_0/random_states_0.pkl
DONE accelerator save !  ==================================================