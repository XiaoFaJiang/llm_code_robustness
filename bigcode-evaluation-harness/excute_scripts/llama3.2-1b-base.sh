conda activate adv
export CUDA_VISIBLE_DEVICES=11
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=llama3.2 --model_type=causal_base --model_name=llama3.2-1b-base --model_path=/data1/model/llama3/meta-llama/Llama-3.2-1B\
  --tasks=mbpp_generate_python_robust_rename


python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=cpp --model_name=llama3.2-1b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=llama3.2-1b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=javascript --model_name=llama3.2-1b-base --perturbation=code_style --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-base --perturbation=rename --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-base --perturbation=code_stmt_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-base --perturbation=code_expression_exchange --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-base --perturbation=insert --model_type=causal_base

python calculate_pass_drop.py --language=python --model_name=llama3.2-1b-base --perturbation=code_style --model_type=causal_base


