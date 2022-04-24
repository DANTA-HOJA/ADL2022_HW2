# --debug --debug_max_sample 會只使用一小部分 dataset 做測試，預設為 100 筆資料

# export DATASET_NAME=swag
export DATASET_PATH=MC_SaveDir

accelerate launch run_swag_no_trainer.py \
  --context_file ${1} \
  --train_file ${2} \
  --validation_file ${3} \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --max_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --checkpointing_steps "epoch" \
  --output_dir ./tmp/$DATASET_PATH/ \
  # --debug --debug_max_sample 10 \
  # --dataset_name $DATASET_NAME \