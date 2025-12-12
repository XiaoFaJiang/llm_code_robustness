conda activate adv

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=insert --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=insert --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=insert --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=insert --model_type=causal_base --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=rename --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=rename --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=rename --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=rename --model_type=causal_base --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_style --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_style --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_style --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_style --model_type=causal_base --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_expression_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_expression_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_expression_exchange --model_type=causal_base --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-7b-base-input-preprocess --perturbation=code_expression_exchange --model_type=causal_base --prompt_type=_preprocess_normalize
