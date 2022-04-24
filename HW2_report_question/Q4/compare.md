# Q4：Pretrained vs Not Pretrained 

## Same model config

```
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
```

## Same args
```
accelerate launch run_qa_no_trainer.py \
  --train_file ./Dataset/train.json \
  --validation_file ./Dataset/valid.json \
  --test_file ./QA_sheet.json \
  --context_file ./Dataset/context.json \
  --model_name_or_path bert-base-chinese \
  --max_seq_length 384 \
  --doc_stride 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --checkpointing_steps "epoch" \
  --output_dir ./tmp/$DATASET_PATH/ \
  --prediction_csv_dir ./prediction.csv \
```

## Without pretrain

```
total epochs = 1 
total Dataloader load times(step) = 38143
total Optimizer update times（step / gradient_accumulation_step） = 19072


epoch = 0

=> train:
    epoch_loss = 3.8207039135621215
    epoch_acc = 2.723959835356422
    epoch_best_acc = 2.723959835356422, epoch_best_loss = 3.8207039135621215 

=> eval（per epoch）:
    epoch_acc = 4.453306746427384
    epoch_best_acc = 4.453306746427384
```


## With pretrain

```
total epochs = 1 
total Dataloader load times(step) = 38143
total Optimizer update times（step / gradient_accumulation_step） = 19072


epoch = 0

=> train:
    epoch_loss = 0.7443623440900035
    epoch_acc = 48.092703772645045
    epoch_best_acc = 48.092703772645045, epoch_best_loss = 0.7443623440900035 

=> eval（per epoch）:
    epoch_acc = 75.54004652708541
    epoch_best_acc = 75.54004652708541
```

## 結論：

沒有 pretrian 基本上 train 不起來，全部設定皆與 "with pretrain" 的一樣，但是 loss 和 accuracy 天差地遠，不知道要 train 多少才可能和有 "with pretrain" 一樣（好像也不太可能因為看圖感覺 loss 已經卡住了）