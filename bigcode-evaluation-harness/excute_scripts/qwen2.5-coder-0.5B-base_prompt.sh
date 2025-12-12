conda activate adv
export CUDA_VISIBLE_DEVICES=2
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=qwen2.5-coder-0.5b-base-prompt --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B \
  --tasks=mbpp_generate_cpp_robust_no_change_1random_prompt,mbpp_generate_cpp_robust_no_change_3random_prompt,mbpp_generate_cpp_robust_no_change_5random_prompt,mbpp_generate_cpp_robust_code_stmt_exchange_1random_prompt,mbpp_generate_cpp_robust_code_stmt_exchange_3random_prompt,mbpp_generate_cpp_robust_code_stmt_exchange_5random_prompt,\
mbpp_generate_python_robust_no_change_1random_prompt,mbpp_generate_python_robust_no_change_3random_prompt,mbpp_generate_python_robust_no_change_5random_prompt,mbpp_generate_python_robust_code_stmt_exchange_1random_prompt,mbpp_generate_python_robust_code_stmt_exchange_3random_prompt,mbpp_generate_python_robust_code_stmt_exchange_5random_prompt,\
mbpp_generate_java_robust_no_change_1random_prompt,mbpp_generate_java_robust_no_change_3random_prompt,mbpp_generate_java_robust_no_change_5random_prompt,mbpp_generate_java_robust_code_stmt_exchange_1random_prompt,mbpp_generate_java_robust_code_stmt_exchange_3random_prompt,mbpp_generate_java_robust_code_stmt_exchange_5random_prompt,\
mbpp_generate_javascript_robust_no_change_1random_prompt,mbpp_generate_javascript_robust_no_change_3random_prompt,mbpp_generate_javascript_robust_no_change_5random_prompt,mbpp_generate_javascript_robust_code_stmt_exchange_1random_prompt,mbpp_generate_javascript_robust_code_stmt_exchange_3random_prompt,mbpp_generate_javascript_robust_code_stmt_exchange_5random_prompt


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_1random_prompt

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_3random_prompt

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_5random_prompt


python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_1random_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_3random_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_5random_prompt


python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_1random_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_3random_prompt

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_5random_prompt


python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_1random_prompt

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_3random_prompt

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-0.5b-base-prompt --perturbation=code_stmt_exchange --model_type=causal_base --prompt_type=_5random_prompt

