conda activate adv
export CUDA_VISIBLE_DEVICES=4,5,6,7
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen1.5 --model_type=causal_base --model_name=codeQwen1.5-7B-base --model_path=/data1/ljc/code/llm_robustness_eval_and_enhance/model/qwen/Qwen/CodeQwen1.5-7B\
  --tasks=mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_insert,mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename,mbpp_generate_python_robust_code_stmt_exchange,\
mbpp_generate_python_robust_code_expression_exchange,mbpp_generate_python_robust_insert,\
mbpp_generate_python_robust_code_style

python calculate_pass_drop.py --language=python --model_name=codeQwen1.5-7B-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codeQwen1.5-7B-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codeQwen1.5-7B-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codeQwen1.5-7B-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=codeQwen1.5-7B-base --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=cpp --model_name=codeQwen1.5-7B-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codeQwen1.5-7B-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codeQwen1.5-7B-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codeQwen1.5-7B-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=codeQwen1.5-7B-base --perturbation=code_style --model_type=causal_base




python calculate_pass_drop.py --language=java --model_name=codeQwen1.5-7B-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codeQwen1.5-7B-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codeQwen1.5-7B-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codeQwen1.5-7B-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=codeQwen1.5-7B-base --perturbation=code_style --model_type=causal_base



python calculate_pass_drop.py --language=javascript --model_name=codeQwen1.5-7B-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codeQwen1.5-7B-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codeQwen1.5-7B-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codeQwen1.5-7B-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=codeQwen1.5-7B-base --perturbation=code_style --model_type=causal_base