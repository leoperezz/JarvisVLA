#! /bin/bash

cuda_visible_devices=4,5,6,7
card_num=4
model_name_or_path="/path/to/your/model/directory"

CUDA_VISIBLE_DEVICES=$cuda_visible_devices vllm serve $model_name_or_path \
    --port 9012 \
    --max-model-len 8448 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size $card_num \
    --trust-remote-code \
    --served_model_name "actvlp" \
    --limit-mm-per-prompt image=5 \
    #--dtype "float32" \
    #--kv-cache-dtype "fp8" \
