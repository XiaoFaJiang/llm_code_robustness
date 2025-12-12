conda activate adv
export CUDA_VISIBLE_DEVICES=11,12
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-7b-instruct --model_path=/data1/model/qwen/Qwen/Qwen2.5-7B-Instruct\
  --tasks=mbpp_generate_python_robust_rename_instruct,\
mbpp_generate_python_robust_code_stmt_exchange_instruct,mbpp_generate_python_robust_code_expression_exchange_instruct,\
mbpp_generate_python_robust_insert_instruct,mbpp_generate_python_robust_code_style_instruct,\
mbpp_generate_javascript_robust_rename_instruct,mbpp_generate_javascript_robust_insert_instruct,\
mbpp_generate_javascript_robust_code_style_instruct
 #跑完了