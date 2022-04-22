# ADL2022_HW2
110.2 CSIE5431_深度學習之應用（ADL）

# Environment settings

Follow [README.md](https://github.com/huggingface/transformers/tree/main/examples) to install packages using by huggingface/transformers.

With TA's settings，pytorch version will be ```1.10.1+cu111```，can install using
    
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# Task 1：Context Selection

Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/multiple-choice

Using ```run_swag_no_trainer.py```

# Task 2：Question Answering
Sample code：https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

Using ```run_qa_no_trainer.py```

# NOTE for Multiple Choice

[[NOTE] Multiple Choice]()
[[NOTE] Question Answering]()

☆ before submit change ```torch.cuda.set_device(1)``` -> ```torch.cuda.set_device(0) ``` in both file（```run_swag_no_trainer.py```, ```run_qa_no_trainer.py```）, because there is only one graphic card exist.


# Question2
```
☆ Task 1： Multiple Choice 
    - yourmodel（model_config_file）：　
    - performance：
    - loss function：Nllloss()
    - opimization：
    - learning rate：
    - batch_size：
```
```
☆ Task 2：Question Answering
    - yourmodel（model_config_file）：　
    - performance：
    - loss function：Nllloss()
    - opimization：
    - learning rate：
    - batch_size：
```
