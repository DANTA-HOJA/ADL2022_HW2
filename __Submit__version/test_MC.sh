# export DATASET_PATH=MC_SaveDir

accelerate launch test_MC.py \
  --context_file ${1} \
  --test_file ${2} \
  --predict_file ./QA_sheet.json \
  --model_name_or_path ./MC_task/20220424_0020AM_09621/MC_SaveDir \
  --per_device_test_batch_size 16 \
  # --train_file ./Dataset/train.json \
  # --validation_file ./Dataset/valid.json \
  # --max_length 512 \
  # --per_device_train_batch_size 2 \
  # --gradient_accumulation_steps 4 \
  # --num_train_epochs 1 \
  # --learning_rate 3e-5 \
  # --checkpointing_steps "epoch" \
  # --output_dir ./tmp/$DATASET_PATH/ \
  # --debug --debug_max_sample 10 \
  # --dataset_name $DATASET_NAME \