#!/bin/bash

base_url=http://localhost:9012/v1
workers=1
max_frames=400
temperature=0.9
history_num=4
action_chunk_len=1
instruction_type="recipe"
model_local_path="/mc-vla-qwen2-vl-7b-250315-A800-c32-e1-b4-a1" #"mc-vla-qwen2-vl-7b-250311-A800-c32-e1-b16-a1"

tasks=(
    smelt/iron_ingot
    craft/craft_crafting_table
)

for checkpoint in  107 ; do  #800 900 1000 1100 1300
    echo "Running for checkpoint $checkpoint..."

    checkpoint_path="/public/JARVIS/checkpoints2/$model_local_path/checkpoint-$checkpoint"
    model_name_or_path="/public/JARVIS/checkpoints2/$model_local_path/checkpoint-$checkpoint"
    log_path_name="$model_local_path-$checkpoint-$env_file"

    for task in "${tasks[@]}"; do
        env_config="$task"

        # Evaluate
        num_iterations=$(($workers / 5 + 1))
        for ((i = 0; i < num_iterations; i++)); do
            python actvlp/evaluate/evaluate.py \
                --workers $workers \
                --env-config $env_config \
                --max-frames $max_frames \
                --temperature $temperature \
                --checkpoints $model_name_or_path \
                --video-main-fold "/home/lmy/workspace/ACTVLP/logs/" \
                --base-url "$base_url" \
                --history-num $history_num \
                --instruction-type $instruction_type \
                --action-chunk-len $action_chunk_len \
                #--verbos True \
            # 如果 Python 脚本执行成功，则退出循环
            if [[ $? -eq 0 ]]; then
                echo "第 $i 次迭代中的 Python 脚本执行成功，退出循环。"
                break
            fi
            if [[ $i -lt $((num_iterations - 1)) ]]; then
                echo "等待 10 秒..."
                sleep 10
            fi
        done
    done 
done 