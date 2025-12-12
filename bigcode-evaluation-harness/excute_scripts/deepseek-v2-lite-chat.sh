conda activate adv
export CUDA_VISIBLE_DEVICES=5,6,7,8
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_type=causal_chat --model_series=deepseek-v2 --model_name=deepseek-v2-lite-chat --model_path=/data1/model/deepseek/deepseek-ai/DeepSeek-V2-Lite-Chat \
  --trust_remote_code \
  --tasks=mbpp_generate_java_robust_rename_instruct,mbpp_generate_java_robust_code_stmt_exchange_instruct,\
mbpp_generate_java_robust_code_expression_exchange_instruct,mbpp_generate_java_robust_insert_instruct,mbpp_generate_java_robust_code_style_instruct

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-chat --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-chat --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-chat --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-chat --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=deepseek-v2-lite-chat --perturbation=code_style --model_type=causal_chat