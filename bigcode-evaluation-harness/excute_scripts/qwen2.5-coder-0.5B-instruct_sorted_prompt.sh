conda activate adv
export CUDA_VISIBLE_DEVICES=9,10
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-0.5b-instruct-prompt --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --tasks=mbpp_generate_javascript_robust_rename_instruct_3random_prompt,mbpp_generate_javascript_robust_rename_instruct_5random_prompt


python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5sorted_prompt

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_5random_prompt


python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=code_style --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=insert --model_type=causal_chat --prompt_type=_5sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-instruct-prompt --perturbation=no_change --model_type=causal_chat --prompt_type=_5sorted_prompt