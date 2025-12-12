conda activate adv
export CUDA_VISIBLE_DEVICES=10
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=llama3.2 --model_type=causal_chat --model_name=llama3.2-1b-instruct --model_path=/data1/model/llama3/meta-llama/Llama-3.2-1B-Instruct\
  --tasks=mbpp_generate_python_robust_no_change_instruct,mbpp_generate_python_robust_rename_instruct,mbpp_generate_python_robust_insert_instruct,mbpp_generate_python_robust_code_style_instruct
#正在跑

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-instruct --perturbation=code_style --model_type=causal_chat


