conda activate adv
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=llama3.1 --model_type=causal_base --model_name=llama3.1-8b-base --model_path=/data1/model/llama3/meta-llama/Llama-3.1-8B\
  --tasks=mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename,mbpp_generate_python_robust_code_stmt_exchange,\
mbpp_generate_python_robust_code_expression_exchange,mbpp_generate_python_robust_insert,\
mbpp_generate_python_robust_code_style

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.1-8b-base --perturbation=code_style --model_type=causal_base