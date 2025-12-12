conda activate adv
export CUDA_VISIBLE_DEVICES=5,6,7,8
python main.py  --allow_code_execution --save_generations --precision=bf16 --model_type=casual_chat --model_name=deepseek-v2-lite-instruct --model_path=/data1/model/deepseek/deepseek-ai/DeepSeek-V2-Lite-Chat \
  --trust_remote_code \
  --tasks=mbpp_generate_java_robust_rename_instruct
#跑完了