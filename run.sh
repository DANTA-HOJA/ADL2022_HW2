echo -e "context file <= ${1}"
echo -e "test file <= ${2}"
echo -e "save to ${3}"

unzip ./__Submit__test_version/MC_task.zip
unzip ./__Submit__test_version/QA_task.zip
bash ./__Submit__test_version/test_MC.sh ${1} ${2}
bash ./__Submit__test_version/test_QA.sh ${1} ./QA_sheet.json ${3}

echo -e "process complete"