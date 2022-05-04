# HOW TO TRAIN

MC task：

    bash run_MC_submit_ver.sh {path/to/context.json} {path/to/train.json} {path/to/valid.json}

QA task：

    bash run_QA_submit_ver.sh {path/to/context.json} {path/to/train.json} {path/to/valid.json}

# HOW TO PLOT（QA only）

After training, ```training_logs.json``` will be generate under ```./tmp/QA_SaveDir/```, using

    bash plot ./tmp/QA_SaveDir/training_logs.json

to plot, and figure will generate at current directory（file extension==```png```）
