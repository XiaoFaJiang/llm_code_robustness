conda activate adv
export CUDA_VISIBLE_DEVICES=4,5,6,7,8
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_type=causal_base --model_series=deepseek-v2 --model_name=deepseek-v2-lite-base --model_path=/data1/model/deepseek/deepseek-ai/DeepSeek-V2-Lite\
  --trust_remote_code\
  --tasks=mbpp_generate_java_robust_rename,mbpp_generate_java_robust_insert,mbpp_generate_cpp_robust_no_change,mbpp_generate_cpp_robust_rename,mbpp_generate_cpp_robust_code_stmt_exchange,\
mbpp_generate_cpp_robust_code_expression_exchange,mbpp_generate_cpp_robust_insert,mbpp_generate_cpp_robust_code_style



python calculate_pass_drop.py --language=cpp --model_name=deepseek-v2-lite-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=deepseek-v2-lite-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=deepseek-v2-lite-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=deepseek-v2-lite-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=deepseek-v2-lite-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-base --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=javascript --model_name=deepseek-v2-lite-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=deepseek-v2-lite-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=deepseek-v2-lite-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=deepseek-v2-lite-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=deepseek-v2-lite-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=deepseek-v2-lite-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=deepseek-v2-lite-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=deepseek-v2-lite-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=deepseek-v2-lite-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=deepseek-v2-lite-base --perturbation=code_style --model_type=causal_base
#跑完了 