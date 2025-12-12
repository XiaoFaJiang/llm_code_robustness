python3 apply_pipeline.py --lang=python --file=mbpp_python_completion.xlsx --code_col_name=code_str_deleted
python3 apply_pipeline.py --lang=cpp --file=mbpp_cpp_completion.xlsx --code_col_name=code_str_deleted
python3 apply_pipeline.py --lang=java --file=mbpp_java_completion.xlsx --code_col_name=code_str_deleted
python3 apply_pipeline.py --lang=javascript --file=mbpp_javascript_completion.xlsx --code_col_name=code_str_deleted
mv mbpp_python_tested.xlsx /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/excel2json/mbpp_python_tested.xlsx
mv mbpp_cpp_tested.xlsx /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/excel2json/mbpp_cpp_tested.xlsx
mv mbpp_java_tested.xlsx /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/excel2json/mbpp_java_tested.xlsx
mv mbpp_javascript_tested.xlsx /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/excel2json/mbpp_javascript_tested.xlsx
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/excel2json/
source bash.sh
mv mbpp_python_completion_tested.json /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/complete/codebert/mbpp_python_completion_tested.json
mv mbpp_cpp_completion_tested.json /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/complete/codebert/mbpp_cpp_completion_tested.json
mv mbpp_javascript_completion_tested.json /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/complete/codebert/mbpp_javascript_completion_tested.json
mv mbpp_java_completion_tested.json /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/complete/codebert/mbpp_java_completion_tested.json