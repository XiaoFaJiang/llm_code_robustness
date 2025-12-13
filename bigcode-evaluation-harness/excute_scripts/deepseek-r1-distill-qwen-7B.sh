conda activate adv
export CUDA_VISIBLE_DEVICES=5,6,7,8,9,10
python main.py  --allow_code_execution --save_generations --max_new_tokens=2048 --precision=bf16 --model_series=deepseek-r1 --model_type=causal_chat --model_name=deepseek-r1-distill-qwen-7B --model_path=/data1/ljc/code/llm_robustness_eval_and_enhance/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B\
  --tasks=mbpp_generate_python_robust_no_change_instruct