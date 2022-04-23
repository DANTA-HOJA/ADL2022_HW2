#--max_train_samples 50 -> 切 50 筆 train_data 測試
# --max_eval_samples 50 -> 切 50 筆 eval_data 測試
# --max_predict_samples 10 -> 切 10 筆 test_data 測試

# export DATASET_NAME=squad
# export DATASET_PATH=QA_SaveDir

accelerate launch test_QA.py \
  --context_file ${1} \
  --test_file ${2} \
  --prediction_csv_dir ${3} \
  --model_name_or_path ./QA_task/20220423_1109PM_07856/QA_SaveDir \
  --per_device_test_batch_size 16 \
  



  # --test_file ./QA_sheet.json \
  # --train_file ./Dataset/train.json \
  # --validation_file ./Dataset/valid.json \
  # --context_file ./Dataset/context.json \
  # --model_name_or_path hfl/chinese-roberta-wwm-ext \
  # --max_seq_length 512 \
  # --doc_stride 128 \
  # --per_device_train_batch_size 2 \
  # --gradient_accumulation_steps 4 \
  # --num_train_epochs 1 \
  # --learning_rate 3e-5 \
  # --checkpointing_steps "epoch" \
  # --output_dir ./tmp/$DATASET_PATH/ \
  # --max_train_samples 50 \
  # --max_eval_samples 50 \
  # --max_predict_samples 10 \
  #--dataset_name $DATASET_NAME \
  
  
 
  # --model_name_or_path bert-base-chinese \

  # --train_file ./Dataset/train.json \
  # --validation_file ./Dataset/valid.json \
  # --test_file ./Dataset/test.json \

  # --- For predition ---
  # --max_answer_length -> default=30
  # --n_best_size -> default=20