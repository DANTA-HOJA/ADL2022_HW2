# MC task

☆ Default format：
```
export DATASET_PATH=MC_SaveDir

accelerate launch run_swag_no_trainer.py \
  --train_file ./Dataset/train.json \
  --validation_file ./Dataset/valid.json \
  --context_file ./Dataset/context.json \
  --model_name_or_path bert-base-chinese \
  --max_length 384 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --checkpointing_steps "epoch" \
  --output_dir ./tmp/$DATASET_PATH/ \
```
```
- 20220423_0405AM_093 -> NO_CHANGE
```
```
- 20220423_0635AM_095 -> max_length 512
```
```
- 20220423_0801PM_095 -> 
    --model_name_or_path hfl/chinese-bert-wwm-ext
    --max_length 512
```
```
- 20220424_0020_09621 -> 
    --model_name_or_path hfl/chinese-roberta-wwm-ext,
    --max_seq_length 512
    --per_device_train_batch_size 2,
    --gradient_accumulation_steps 4,
```

# QA task

☆ Default format：
```
export DATASET_PATH=QA_SaveDir

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

```
- 20220423_0530AM_075 -> NO_CHANGE
```
```
- 20220423_0628PM_075 -> --max_seq_length 512
```
```
- 20220423_0900PM_076 -> 
    --model_name_or_path hfl/chinese-bert-wwm-ext,
    --max_seq_length 512,
```
```
- 20220423_0953PM_07687  -> 
    --model_name_or_path hfl/chinese-bert-wwm-ext,
    --per_device_train_batch_size 2,
    --gradient_accumulation_steps 4,
```
```
- 20220423_1109PM_07856  -> 
    --model_name_or_path hfl/chinese-roberta-wwm-ext,
    --max_seq_length 512
    --per_device_train_batch_size 2,
    --gradient_accumulation_steps 4,
```
```
- 20220425_0200AM_07319  -> 
    --model_name_or_path hfl/chinese-roberta-wwm-ext,
    --max_seq_length 512
    --per_device_train_batch_size 4,
    --num_train_epochs 2,
```
```
- 20220425_0309AM  -> default setting but no pretrain
```