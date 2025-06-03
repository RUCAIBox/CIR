#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# 定义 target_path 的列表
target_paths=(
    # "Model Path"
)

for target_path in "${target_paths[@]}"; do
    echo "Running evaluation with target_path: $target_path"

    # 执行 Python 脚本
    python ./run.py \
        --data_name AIME24 \
        --target_path "$target_path" \
        --model_name_or_path "$target_path" \
        --max_tokens 16384 \
        --paralle_size 1 \
        --decode sample \
        --n 16 \
        --prompt_template CIR-qwen-math \
        --prompt code_r1 \
        --exe_code

    # 检查上一步的退出状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing target_path: $target_path"
        exit 1
    fi

    echo "Finished evaluation for target_path: $target_path"
done

echo "All evaluations completed."

for target_path in "${target_paths[@]}"; do
    echo "Running evaluation with target_path: $target_path"

    # 执行 Python 脚本
    python ./run.py \
        --data_name AIME25 \
        --target_path "$target_path" \
        --model_name_or_path "$target_path" \
        --max_tokens 16384 \
        --paralle_size 1 \
        --decode sample \
        --n 16 \
        --prompt_template CIR-qwen-math \
        --prompt code_r1 \
        --exe_code

    # 检查上一步的退出状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing target_path: $target_path"
        exit 1
    fi

    echo "Finished evaluation for target_path: $target_path"
done

echo "All evaluations completed."

for target_path in "${target_paths[@]}"; do
    echo "Running evaluation with target_path: $target_path"

    # 执行 Python 脚本
    python ./run.py \
        --data_name AMC23 \
        --target_path "$target_path" \
        --model_name_or_path "$target_path" \
        --max_tokens 16384 \
        --paralle_size 1 \
        --decode sample \
        --n 16 \
        --prompt_template CIR-qwen-math \
        --prompt code_r1 \
        --exe_code

    # 检查上一步的退出状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing target_path: $target_path"
        exit 1
    fi

    echo "Finished evaluation for target_path: $target_path"
done

echo "All evaluations completed."

for target_path in "${target_paths[@]}"; do
    echo "Running evaluation with target_path: $target_path"

    # 执行 Python 脚本
    python ./run.py \
        --data_name MATH_OAI \
        --target_path "$target_path" \
        --model_name_or_path "$target_path" \
        --max_tokens 16384 \
        --paralle_size 1 \
        --decode greedy \
        --n 1 \
        --prompt_template CIR-qwen-math \
        --prompt code_r1 \
        --exe_code

    # 检查上一步的退出状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing target_path: $target_path"
        exit 1
    fi

    echo "Finished evaluation for target_path: $target_path"
done

echo "All evaluations completed."

for target_path in "${target_paths[@]}"; do
    echo "Running evaluation with target_path: $target_path"

    # 执行 Python 脚本
    python ./run.py \
        --data_name olymmath-easy-100 \
        --target_path "$target_path" \
        --model_name_or_path "$target_path" \
        --max_tokens 16384 \
        --paralle_size 1 \
        --decode sample \
        --n 16 \
        --prompt_template CIR-qwen-math \
        --prompt code_r1 \
        --exe_code

    # 检查上一步的退出状态码
    if [ $? -ne 0 ]; then
        echo "Error occurred while processing target_path: $target_path"
        exit 1
    fi

    echo "Finished evaluation for target_path: $target_path"
done

echo "All evaluations completed."