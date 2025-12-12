conda activate adv
export CUDA_VISIBLE_DEVICES=9,10,11,12,13
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=llama3.1 --model_type=causal_chat --model_name=llama3.1-8b-instruct --model_path=/data1/model/llama3/meta-llama/Llama-3.1-8B-Instruct\
  --tasks=mbpp_generate_python_robust_no_change_instruct,mbpp_generate_python_robust_rename_instruct
#正在跑

python calculate_pass_drop.py --language=cpp --model_name=llama3.1-8b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.1-8b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.1-8b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.1-8b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=llama3.1-8b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.1-8b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.1-8b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.1-8b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.1-8b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=llama3.1-8b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.1-8b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.1-8b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.1-8b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.1-8b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=javascript --model_name=llama3.1-8b-instruct --perturbation=code_style --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-instruct --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-instruct --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-instruct --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-instruct --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-instruct --perturbation=code_style --model_type=causal_chat


