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


ending_names = ['ending0', 'ending1', 'ending2', 'ending3']
context_name = sent1
quetion_header_name = sent2
label_column_name = label
raw_datasets['train'] = 
Dataset({
    features: ['video-id', 'fold-ind', 'startphrase', 'sent1', 'sent2', 'gold-source', 'ending0', 'ending1', 'ending2', 'ending3', 'label'],
    num_rows: 73546
})


tokenizer_input_format{
    context = Someone stares blearily down at the floor.
    question_headers = Someone
    first_sentences = ['Someone stares blearily down at the floor.', 'Someone stares blearily down at the floor.', 'Someone stares blearily down at the floor.', 'Someone stares blearily down at the floor.']
    question_headers = Someone
    second_sentences = ["Someone shifts his amused gaze to someone's brow pinched uneasily.", 'Someone takes the gun and leads his men upstairs into the kitchen.', 'Someone looks up at someone in shock.', "Someone takes the bag, revealing a sleek silver - diamond shot at someone's head."]
    labels = 2
}

{'id': '593f14f960d971e294af884f0194b3a7', 'question': '舍本和誰的數據能推算出連星的恆星的質量？', \
    'paragraphs': [2018, 6952, 8264, 836], 'relevant': 836, 'answer': {'text': '斯特魯維', 'start': 108}}

question_name = "question"
context_names = [f"context_{i}" for i in range(4)]
relevant_name = "relevant"

print(f"type(raw_datasets['train']) = {type(raw_datasets['train'])}")
print(f"context_name = {context_name}")
print(f"question_name = {question_name}")
print(f"relevant_name = {relevant_name}")
print(f"raw_datasets['train'] = \n{raw_datasets['train'][890]}\n")

{
    'id': '3ccf8146272ad3f639e42ef9f67fa2a7', 
    'question': '宇宙正在膨脹是根據誰的定律?',
    'context_0': 92,
    'context_1': 6751,
    'context_2': 2355,
    'context_3': 7288,
    'relevant': 6751,
    'answer_text': '哈伯定律',
    'answer_start': 26
}


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