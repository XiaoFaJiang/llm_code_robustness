conda activate adv
export CUDA_VISIBLE_DEVICES=6,7,8,9,10
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=qwen2.5-coder-3b-base-input-preprocess --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-3B \
  --tasks=mbpp_generate_python_robust_rename_preprocess_normalize