conda activate adv
export CUDA_VISIBLE_DEVICES=0
python ../main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-0.5b-instruct-english-prompt --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --tasks=mbpp_generate_python_robust_no_change_instruct,mbpp_generate_cpp_robust_no_change_instruct\
,mbpp_generate_java_robust_no_change_instruct,mbpp_generate_javascript_robust_no_change_instruct\
,mbpp_generate_python_robust_rename_instruct,mbpp_generate_cpp_robust_rename_instruct\
,mbpp_generate_java_robust_rename_instruct,mbpp_generate_javascript_robust_rename_instruct
#跑完了   
