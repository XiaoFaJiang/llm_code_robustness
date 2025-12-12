conda activate adv
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,13,14
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=codellama --model_type=causal_chat --model_name=codellama-7b-instruct --model_path=/data1/model/llama2/codellama/CodeLlama-7b-Instruct-hf\
  --tasks=mbpp_generate_javascript_robust_no_change_instruct,mbpp_generate_javascript_robust_rename_instruct,mbpp_generate_javascript_robust_code_stmt_exchange_instruct,\
mbpp_generate_javascript_robust_code_expression_exchange_instruct,mbpp_generate_javascript_robust_insert_instruct,mbpp_generate_javascript_robust_code_style_instruct
#正在跑

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-instruct --perturbation=code_style --model_type=causal_chat