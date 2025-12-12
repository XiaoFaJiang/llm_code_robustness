conda activate adv
export CUDA_VISIBLE_DEVICES=7,8,9,10
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_type=causal_base --model_series=deepseek-v2 --model_name=deepseek-coder-v2-lite-base --model_path=/data1/model/deepseek/deepseek-ai/DeepSeek-Coder-V2-Lite-Base\
  --trust_remote_code\
  --tasks=mbpp_generate_java_robust_rename,mbpp_generate_java_robust_code_expression_exchange,mbpp_generate_java_robust_code_stmt_exchange

python calculate_pass_drop.py --language=java --model_name=deepseek-coder-v2-lite-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-coder-v2-lite-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-coder-v2-lite-base --perturbation=code_stmt_exchange --model_type=causal_base
