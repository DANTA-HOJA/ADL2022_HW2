# ADL2022_HW2
110.2 CSIE5431_深度學習之應用（ADL）

# Environment settings

Follow [README.md](https://github.com/huggingface/transformers/tree/main/examples) to install packages used in ```huggingface/transformers```.


- python == 3.8
- torch == 1.10.1
- accelerate == 0.6.2
> Warning： with ```accelerate==0.6.2``` doing ```Accelerator(log_with="all", logging_dir=args.output_dir)``` will cause error, upgrade to ```accelerate==0.7.1``` to fix）


With TA's settings，torch version will be ```1.10.1+cu111```， can install by
    
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html


# Task 1：Context Selection

Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice

Modified using ```run_swag_no_trainer.py```

# Task 2：Question Answering
Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

Modified using ```run_qa_no_trainer.py```

# NOTE for Multiple Choice

[[NOTE] Multiple Choice](./__NOTE__MC_PARACONFIG.py)

[[NOTE] Question Answering](./__NOTE__QA_PARACONFIG.py)

☆ Before submit, change ```torch.cuda.set_device(1)``` -> ```torch.cuda.set_device(0) ``` in both files（```run_swag_no_trainer.py```, ```run_qa_no_trainer.py```）, because there is only one graphic card exist.


# Report

[Submit Report](./__Submit__version/report.pdf)
