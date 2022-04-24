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



Log file <= Training_Histroy/QA_task/20220423_0530AM_075/QA_SaveDir/training_logs.json

==================================================================================================== 
=> training_logs.keys()
dict_keys(['epoch', 'cum_avg_batch_loss', 'epoch_loss', 'epoch_best_loss', 'cum_avg_batch_acc', 'epoch_acc', 'epoch_best_acc', 'total_completed_steps (optimizer update)'])
dict_keys(['epoch', 'epoch_acc', 'epoch_best_acc']) 

==================================================================================================== 
total epochs = 1 

total Dataloader load times(step) = 38143
total Optimizer update times（step / gradient_accumulation_step） = 19072
==================================================================================================== 
epoch = 0
=> train:
    epoch_loss = 0.7443623440900035
    epoch_acc = 48.092703772645045
    epoch_best_acc = 48.092703772645045, epoch_best_loss = 0.7443623440900035 

=> eval（per epoch）:
    epoch_acc = 75.54004652708541
    epoch_best_acc = 75.54004652708541