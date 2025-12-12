conda activate adv
export CUDA_VISIBLE_DEVICES=6,7
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_series=qwen2.5 --model_type=causal_base --model_name=Qwen2.5-coder-7b-base --model_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-7B --tasks=mbpp_generate_python_robust_rename
#已经全部跑完