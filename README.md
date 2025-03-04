Official Implementation of Paper "Learning to Better Act by Post-training on Vision Language Tasks"

The code will be released soon (in a week) ...

## Installation
- Install dependencies.
```shell
git clone https://github.com/CraftJarvis/ActVLP.git
conda create -n actvlp python=3.10
conda activate actvlp
cd ActVLP
pip install -e .
```

- If you want to evaluate ActVLP model, please download [Minestudio](https://github.com/CraftJarvis/MineStudio), an open-source software package in Minecraft(the following commands are copy from Minestudio).

```shell
conda install --channel=conda-forge openjdk=8 -y
# MineStudio is available on PyPI. You can install it via pip.
pip install MineStudio
# After the installation, you can run the following command to check if the installation is successful:
python -m minestudio.simulator.entry # using Xvfb
MINESTUDIO_GPU_RENDER=1 python -m minestudio.simulator.entry # using VirtualGL
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