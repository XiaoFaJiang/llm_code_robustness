conda activate adv
export CUDA_VISIBLE_DEVICES=0,1,2
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-3b-instruct-input-preprocess --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-3B-Instruct \
  --tasks=mbpp_generate_python_robust_rename_instruct_preprocess_normalize

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=insert --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=insert --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=insert --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=insert --model_type=causal_chat --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=rename --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=rename --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=rename --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=rename --model_type=causal_chat --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_style --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_style --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_style --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_style --model_type=causal_chat --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_stmt_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=java --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize

python calculate_pass_drop.py --language=javascript --model_name=qwen2.5-coder-3b-instruct-input-preprocess --perturbation=code_expression_exchange --model_type=causal_chat --prompt_type=_preprocess_normalize
