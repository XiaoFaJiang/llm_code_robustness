# 任务 1: mbpp_generate_java_robust_code_stmt_exchange_instruct_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5sorted_prompt

# 任务 1: mbpp_generate_java_robust_code_stmt_exchange_instruct_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_5sorted_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1random_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3random_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5random_prompt


python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_3sorted_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_1sorted_prompt