conda activate adv
export CUDA_VISIBLE_DEVICES=1,14
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=Qwen2.5-coder-1.5b-base-adv-train --peft_model=/data1/ljc/code/llm_robustness_eval_and_enhance/adv_samples_gen/Qwen2.5-Coder-1.5B-Base-LoRA/checkpoint-4398 --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B\
  --tasks=mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename,mbpp_generate_python_robust_code_stmt_exchange,\
mbpp_generate_python_robust_code_expression_exchange,mbpp_generate_python_robust_insert,\
mbpp_generate_python_robust_code_style,\
mbpp_generate_cpp_robust_no_change,mbpp_generate_cpp_robust_rename,mbpp_generate_cpp_robust_code_stmt_exchange,\
mbpp_generate_cpp_robust_code_expression_exchange,mbpp_generate_cpp_robust_insert,mbpp_generate_cpp_robust_code_style,\
mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_code_stmt_exchange,\
mbpp_generate_javascript_robust_code_expression_exchange,mbpp_generate_javascript_robust_insert,\
mbpp_generate_javascript_robust_code_style,\
mbpp_generate_java_robust_no_change,mbpp_generate_java_robust_rename,\
mbpp_generate_java_robust_code_stmt_exchange,mbpp_generate_java_robust_code_expression_exchange,\
mbpp_generate_java_robust_insert,mbpp_generate_java_robust_code_style


python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_style --model_type=causal_base


python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=Qwen2.5-coder-1.5b-base-adv-train --perturbation=code_style --model_type=causal_base