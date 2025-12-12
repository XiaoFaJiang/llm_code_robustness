conda activate adv
export CUDA_VISIBLE_DEVICES=6,7,8,9
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=Qwen2.5-coder-3b-base --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-3B\
  --tasks=mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_insert


python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-3b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-3b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-3b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-3b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-3b-base --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-3b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-3b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-3b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-3b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-3b-base --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-3b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-3b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-3b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-3b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-3b-base --perturbation=code_style --model_type=causal_base