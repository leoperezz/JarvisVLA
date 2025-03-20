Official Implementation of Paper "Learning to Better Act by Post-training on Vision Language Tasks"

The code will be released soon (in a week) ...

## Installation
Install dependencies.
```shell
git clone https://github.com/CraftJarvis/JarvisVLA.git
conda create -n mcvla python=3.10
conda activate mcvla
cd JarvisVLA
conda install --channel=conda-forge openjdk=8 -y
pip install -e .
```

After the installation, you can run the following command to check if the installation is successful and the environment is working:

```shell
# After the installation, you can run the following command to check if the installation is successful:
python -m minestudio.simulator.entry # using Xvfb
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry # using VirtualGL
```

## Rollout 

You can serve the model with vllm to support multi-GPU and multi-process rollout.
```sh
CUDA_VISIBLE_DEVICES=0 vllm serve jarvis_vla_qwen2_vl_7b_sft --port 8000
```

Then you need to edit the rollout script to the use the correct base_url and port. 
Finally, you can run the rollout script.
```sh
sh scripts/evaluate/rollout-kill.sh
```

## Train

Prepare the dataset and base model, and write their locations in the shell below.

- Single GPU
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft.sh
```
- Multi-GPU
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft-multi-GPU.sh
```
- Multi-Node
```shell
sh scripts/vla/vla_qwen2_vl_7b_sft-multi-node.sh
```