#--max_train_samples 50 -> 切 50 筆 train_data 測試
# --max_eval_samples 50 -> 切 50 筆 eval_data 測試
# --max_predict_samples 10 -> 切 10 筆 test_data 測試

echo -e "context file <= ${1}"
echo -e "train file <= ${2}"
echo -e "validation file <= ${3}"

export DATASET_NAME=squad
export DATASET_PATH=QA_SaveDir

accelerate launch run_qa_no_trainer.py \
  --context_file ${1} \
  --train_file ${2} \
  --validation_file ${3} \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 2 \
  --learning_rate 3e-5 \
  --checkpointing_steps "epoch" \
  --output_dir ./tmp/$DATASET_PATH/ \
  --prediction_csv_dir ./prediction.csv \
  # --max_train_samples 50 \
  # --max_eval_samples 50 \
  # --max_predict_samples 10 \
  # --dataset_name $DATASET_NAME \
  
echo -e "QA process complete"
  
 
  # --model_name_or_path bert-base-chinese \

  # --train_file ./Dataset/train.json \
  # --validation_file ./Dataset/valid.json \
  # --test_file ./Dataset/test.json \

  # --- For predition ---
  # --max_answer_length -> default=30
  # --n_best_size -> default=20