conda activate adv
export CUDA_VISIBLE_DEVICES=13
python main.py  --allow_code_execution --save_generations --max_new_tokens=2048 --precision=bf16 --model_series=deepseek-r1 --model_type=causal_chat --model_name=deepseek-r1-distill-qwen-1.5B --model_path=/data1/ljc/code/llm_robustness_eval_and_enhance/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\
  --tasks=mbpp_generate_python_robust_no_change_instruct