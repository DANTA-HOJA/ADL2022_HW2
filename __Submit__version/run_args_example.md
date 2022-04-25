# Train MC

bash ./run_MC_submit_ver.sh ./context.json ./train.json ./valid.json 

# Train QA

bash ./run_QA_submit_ver.sh ./context.json ./train.json ./valid.json 

# MC Task

bash ./test_MC.sh ./context.json ./test.json

# QA Task

bash ./test_QA.sh ./context.json ./QA_sheet.json ./prediction.csv

# Plot

bash ./plot.sh Training_Histroy/MC_task/20220424_0020AM_09621/MC_SaveDir/training_logs.json





# run.sh 

bash ./run.sh ./context.json ./test.json ./prediction.csv