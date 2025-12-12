conda activate adv
export CUDA_VISIBLE_DEVICES=11
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-0.5b-instruct-adv-train --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct --peft_model=/data1/ljc/code/llm_robustness_eval_and_enhance/adv_samples_gen/Qwen2.5-Coder-0.5B-Instruct-LoRA/checkpoint-2199\
  --tasks=mbpp_generate_cpp_robust_rename_instruct,mbpp_generate_cpp_robust_code_expression_exchange_instruct,\
mbpp_generate_cpp_robust_insert_instruct,mbpp_generate_cpp_robust_code_style_instruct


python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-adv-train --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-adv-train --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-adv-train --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-adv-train --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=cpp --model_name=qwen2.5-coder-0.5b-instruct-adv-train --perturbation=code_style --model_type=causal_chat