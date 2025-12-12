conda activate adv
export CUDA_VISIBLE_DEVICES=3,4,5
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-3b-instruct --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-3B-Instruct \
  --tasks=mbpp_generate_python_robust_no_change_instruct,mbpp_generate_python_robust_rename_instruct,\
mbpp_generate_python_robust_code_stmt_exchange_instruct,mbpp_generate_python_robust_code_expression_exchange_instruct,\
mbpp_generate_python_robust_insert_instruct,mbpp_generate_python_robust_code_style_instruct

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct --perturbation=code_style --model_type=causal_chat

