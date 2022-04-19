# export DATASET_NAME=swag
export DATASET_PATH=MC_SaveDir

accelerate launch run_swag_no_trainer.py \
  --train_file ./Dataset/train.json \
  --validation_file ./Dataset/valid.json \
  --context_file ./Dataset/context.json \
  --model_name_or_path bert-base-chinese \
  --max_length 256 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --learning_rate 3e-5 \
  --checkpointing_steps "epoch" \
  --output_dir ./tmp/$DATASET_PATH/ \
  --debug --debug_max_sample 10 \
  # --dataset_name $DATASET_NAME \