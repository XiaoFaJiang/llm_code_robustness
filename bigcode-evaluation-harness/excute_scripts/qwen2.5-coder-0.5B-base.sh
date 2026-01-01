conda activate adv
export CUDA_VISIBLE_DEVICES=2
export OPENAI_BASE_URL='http://210.28.134.32:8800/v1/completions'
export MODEL_ID=Qwen2.5-Coder-0.5B
export concurrency=15
MODEL_NAME=qwen2.5-coder-0.5b-base
MODEL_TYPE=causal_base
MODEL_SERIES=qwen2.5
MODEL_PATH=/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B


python main.py  --api 'http://127.0.0.1:8800/v1/completions' --allow_code_execution \
  --save_generations \
  --precision=bf16 \
  --model_series=qwen2.5 \
  --model_type=${MODEL_TYPE} \
  --model_name=${MODEL_NAME} \
  --model_path=${MODEL_PATH} \
  --tasks=mbpp_generate_python_robust_combined_perturbation,\
mbpp_generate_cpp_robust_combined_perturbation,\
mbpp_generate_java_robust_combined_perturbation,\
mbpp_generate_javascript_robust_combined_perturbation,\
humaneval_generate_python_robust_combined_perturbation,\
humaneval_generate_cpp_robust_combined_perturbation,\
humaneval_generate_java_robust_combined_perturbation,\
humaneval_generate_javascript_robust_combined_perturbation,\
humaneval_generate_python_robust_no_change,humaneval_generate_python_robust_rename,\
humaneval_generate_python_robust_code_stmt_exchange,humaneval_generate_python_robust_code_expression_exchange,\
humaneval_generate_python_robust_insert,humaneval_generate_python_robust_code_style,\
humaneval_generate_cpp_robust_no_change,humaneval_generate_cpp_robust_rename,humaneval_generate_cpp_robust_code_stmt_exchange,\
humaneval_generate_cpp_robust_code_expression_exchange,\
humaneval_generate_cpp_robust_insert,humaneval_generate_cpp_robust_code_style,\
humaneval_generate_javascript_robust_no_change,humaneval_generate_javascript_robust_rename,humaneval_generate_javascript_robust_code_stmt_exchange,\
humaneval_generate_javascript_robust_code_expression_exchange,humaneval_generate_javascript_robust_insert,\
humaneval_generate_javascript_robust_code_style,\
humaneval_generate_java_robust_no_change,humaneval_generate_java_robust_rename,humaneval_generate_java_robust_code_stmt_exchange,\
humaneval_generate_java_robust_code_expression_exchange,humaneval_generate_java_robust_insert,humaneval_generate_java_robust_code_style

# 定义要处理的语言列表
languages=("cpp" "python" "java" "javascript")
perturbations=("combined_perturbation")

# 使用for循环遍历所有语言
for language in "${languages[@]}"; do
    echo "Processing $language..."
    for perturbation in "${perturbations[@]}"; do
        echo "Process $perturbation..."
        python calculate_pass_drop.py \
            --language="$language" \
            --model_name=${MODEL_NAME} \
            --perturbation=${perturbation} \
            --model_type=${MODEL_TYPE}
      echo "Completed processing $language perturbation $perturbation dataset mbpp"
      echo "------------------------------"
    done 
done

# 定义要处理的语言列表
languages=("cpp" "python" "java" "javascript")
perturbations=("combined_perturbation" "code_style" "insert" "rename" "code_stmt_exchange" "code_expression_exchange")

# 使用for循环遍历所有语言
for language in "${languages[@]}"; do
    echo "Processing $language..."
    for perturbation in "${perturbations[@]}"; do
        echo "Process $perturbation..."
        python calculate_pass_drop.py \
            --language="$language" \
            --model_name=${MODEL_NAME} \
            --perturbation=${perturbation} \
            --model_type=${MODEL_TYPE} \
            --dataset='humaneval'
      echo "Completed processing $language perturbation $perturbation dataset humaneval"
      echo "------------------------------"
    done 
done