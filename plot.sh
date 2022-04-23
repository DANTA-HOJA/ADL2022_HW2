echo -e "Log file <= ${1}\n" # -e 配 \n 可以換行
# echo -e "Predict save to => ${2}\n"
# echo -e "Model = intent_cls_8946.ckpt @ ./download_ckpt/intent/"

python3 logs_ploter.py --log_path ${1}