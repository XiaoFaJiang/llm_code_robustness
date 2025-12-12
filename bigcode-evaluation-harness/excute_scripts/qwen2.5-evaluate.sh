conda activate adv
python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=no_change\
  --model_type=causal_base


python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=rename\
  --model_type=causal_base


python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=code_style\
  --model_type=causal_base

python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=code_stmt_exchange\
  --model_type=causal_base


python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=code_expression_exchange\
  --model_type=causal_base

python evaluate_pass_at.py --language=java\
  --model_name=Qwen2.5-coder-0.5b-base\
  --perturbation=insert\
  --model_type=causal_base

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-0.5b-base --perturbation=rename --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-0.5b-base --perturbation=code_stmt_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-0.5b-base --perturbation=code_expression_exchange --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-0.5b-base --perturbation=insert --model_type=causal_chat

python calculate_pass_drop.py --language=java --model_name=Qwen2.5-coder-0.5b-base --perturbation=code_style --model_type=causal_chat