conda activate adv
export CUDA_VISIBLE_DEVICES=6,7,8,9
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=codellama --model_type=causal_base --model_name=codellama-7b-base --model_path=/data1/model/llama2/codellama/CodeLlama-7b-hf\
  --tasks=mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename

python calculate_pass_drop.py --language=python --model_name=codellama-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codellama-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codellama-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codellama-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codellama-7b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codellama-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codellama-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codellama-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codellama-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codellama-7b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codellama-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codellama-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codellama-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codellama-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codellama-7b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codellama-7b-base --perturbation=code_style --model_type=causal_base

