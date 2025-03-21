# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse

[![arXiv](https://img.shields.io/badge/arXiv-2503.16365-df2a2a.svg?style=for-the-badge)](https://arxiv.org/pdf/2503.16365)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/collections/CraftJarvis/jarvis-vla-v1-67dc157a99d011efd7d7f7e4)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)

[**Project Website**](https://craftjarvis.github.io/JarvisVLA/) | [**Datasets**](https://huggingface.co/datasets/CraftJarvis/minecraft-vla-sft) 

## Updates

* [2025.03.21] Our paper can be found in [arXiv](https://arxiv.org/pdf/2503.16365).

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

## Inference 

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

---

### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{li2025jarvisvla,
  title   = {JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse},
  author  = {Muyao Li and Zihao Wang and Kaichen He and Xiaojian Ma and Yitao Liang},
  journal = {arXiv preprint arXiv:2503.16365}, 
  year    = {2025}
}
```
