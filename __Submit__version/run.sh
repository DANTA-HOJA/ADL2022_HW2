echo -e "context file <= ${1}"
echo -e "test file <= ${2}"
echo -e "save to ${3}"

unzip ./MC_task.zip
unzip ./QA_task.zip
bash ./test_MC.sh ${1} ${2}
bash ./test_QA.sh ${1} ./QA_sheet.json ${3}

echo -e "process complete"