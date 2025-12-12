conda activate adv
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_chat --model_name=qwen2.5-coder-7b-instruct-prompt --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-7B-Instruct \
  --tasks=mbpp_generate_python_robust_rename_instruct_1random_prompt,mbpp_generate_python_robust_rename_instruct_3random_prompt,mbpp_generate_python_robust_rename_instruct_5random_prompt,mbpp_generate_python_robust_no_change_instruct_3random_prompt,mbpp_generate_python_robust_no_change_instruct_5random_prompt,\
mbpp_generate_python_robust_rename_instruct_1sorted_prompt,mbpp_generate_python_robust_rename_instruct_3sorted_prompt,mbpp_generate_python_robust_rename_instruct_5sorted_prompt,mbpp_generate_python_robust_no_change_instruct_1sorted_prompt,mbpp_generate_python_robust_no_change_instruct_3sorted_prompt,mbpp_generate_python_robust_no_change_instruct_5sorted_prompt

#!/bin/bashmb

# Python 任务 (random prompts)
# 任务 1-18: Python robust tasks with random prompts
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1random_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3random_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5random_prompt

python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_1sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_3sorted_prompt
python calculate_pass_drop.py --language=python --model_name=qwen2.5-coder-7b-instruct-prompt --perturbation=rename --model_type=causal_chat --prompt_type=_5sorted_prompt
