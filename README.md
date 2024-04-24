# Efficient KV Cache Replacement Policy for LLMs

## Introduction

This repository contains the implementation of the Special-Character-Aware Caching (SCAC) strategy for efficient and effective KV cache eviction in Large Language Models (LLMs), particularly focusing on punctuation management in cached tokens. This project was developed as part of the CS550 Advanced Operating Systems course at Illinois Institute of Technology.

Our approach introduces a novel caching strategy that leverages special characters, particularly punctuations, to improve the efficiency of KV caches in LLMs without sacrificing performance. By optimizing the cache replacement policy, our method aims to balance memory consumption and inference accuracy.

## Project Team

- Haoxuan Wang
- Kaiang Wen
- Project Lead: Xiaoyang Lu

## Features

-**Special-Character-Aware Caching (SCAC):** Targets punctuations for improved caching efficiency.


-**Dynamic Cache Management:** Efficiently manages cache size and eviction policies to optimize performance under hardware constraints.


-**Experimental Validation:** Validated using the Llama-2-7B model with the PG19 dataset, demonstrating improved perplexity scores.

## Experiment Setup

The experiments were conducted under the following settings:

- Model: Llama-2-7B
- Dataset: PG19
- Metric: Perplexity
- Cache size
  - Attention sink: 4
  - Window size + SCAC: 2048
- Hardware: A6000 with 48GB memory

## Usage

### Installation

```bash
git clone git@github.com:glisses/Efficient-Effective-KV-Cache-Replacement-Policy-for-LLMs.git
cd Efficient-Effective-KV-Cache-Replacement-Policy-for-LLMs.git
```

### Environment Setup

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

### Run Chatbot

```bash
python examples/eval_long_ppl_punc.py --model_name_or_path meta-llama/Llama-2b-hf --dataset_name pg19 --split test --enable_Start_recent_kv_cache --enable_pos_shift --start_size 4 --punc_size 128 --recent_size 1920
```

## Results

### **SCAC Size v.s. Performance**

| Attention sink size | Window size | SCAC size | 0.01 * PPL↓ (5.65+) |
| ------------------- | ----------- | --------- | -------------------- |
| 4                   | 2048        | 0         | 0.8117               |
| 4                   | 2044        | 4         | 0.8151               |
| 4                   | 2016        | 32        | 0.5703               |
| 4                   | 1984        | 64        | 0.4689               |
| 4                   | 1920        | 128       | 0.3061               |
| 4                   | 1792        | 256       | 0.387                |
| 4                   | 1536        | 512       | 1.3387               |
| 0                   | 1924        | 128       | N/A                  |
| 4                   | 1892        | 156       | 0.2916               |

![image.png](https://s2.loli.net/2024/04/25/5j1C2c9nXqHAVmv.png)

### **Special character type v.s. Performance**

| Punctuationtype | Attention sink size | Window size | SCAC size | 0.01 * PPL↓ (5.65+) |
| --------------- | ------------------- | ----------- | --------- | -------------------- |
| All             | 4                   | 1920        | 128       | 0.3061               |
| .,?!;:"         | 4                   | 1920        | 128       | 0.3028               |
| .,!?            | 4                   | 1920        | 128       | 0.3745               |
| .?!             | 4                   | 1920        | 128       | 0.5593               |
| .,              | 4                   | 1920        | 128       | 0.3651               |
| ,               | 4                   | 1920        | 128       | 0.3617               |
| .               | 4                   | 1920        | 128       | 0.6093               |

### Ablations

| Corruption type                                           | Attention sink size | Window size | SCAC size | 0.01 * PPL↓ (5.65+) |
| --------------------------------------------------------- | ------------------- | ----------- | --------- | -------------------- |
| None                                                      | 4                   | 1920        | 0         | 1.1835               |
| Randomly corrupt with zero tensors with probability 50%   | 4                   | 1920        | 128       | N/A                  |
| Randomly corrupt with random tensors with probability 50% | 4                   | 1920        | 128       | N/A                  |
| Always replace with the first 128 cached window tokens    | 4                   | 1920        | 128       | 119.6472             |
| Always replace with the last 128 cached window tokens     | 4                   | 1920        | 128       | 1.3843               |

## Acknowledgments

* Special thanks to Prof. Sun and Xiaoyang Lu for guidance and support throughout the project.
* We would like to express our gratitude to [StreamingLLM](https://github.com/mit-han-lab/streaming-llm) for their open-source code, which served as a foundation for our project's codebase.
