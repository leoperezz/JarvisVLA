#! /bin/bash

cuda_visible_devices=0,1,2,3
card_num=4
model_name_or_path="/public/JARVIS/checkpoints2/mc-vla-qwen2-vl-7b-250315-A800-c32-e1-b4-a1/checkpoint-107" #"/path/to/your/model/directory"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port 9052 \
    --max-model-len 8448 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --served_model_name "jarvisvla" \
    --limit-mm-per-prompt image=5 \
    #--dtype "float32" \
    #--kv-cache-dtype "fp8" \
