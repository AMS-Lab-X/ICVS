# Instruction-Conditioned Visual Token Sparsification (ICVS)

Efficient inference for vision-language models (VLMs) via **instruction-conditioned visual token sparsification**. This repository implements the ICVS framework proposed in the paper:

> **Instruction-conditioned visual token sparsification for efficient vision-language model inference**


---

## Highlights

- Reformulates visual token pruning as an **instruction-aware process**.
- Leverages **task-relevant attention heads** to guide token selection.
- Introduces **density-aware token recycling** to preserve complementary visual information.
- Achieves a better accuracy-efficiency trade-off for multimodal reasoning under constrained token budgets.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AMS-Lab-X/ICVS.git
cd ICVS
```

### 2. Create conda environment and install dependencies

```bash
conda create -n ICVS python=3.10 -y
conda activate ICVS

pip install -e .

pip install transformers==4.37.2
pip install flash_attn==2.3.3
```

### 3. Download evaluation benchmarks

Please follow the official LLaVA evaluation instructions:

[LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

---

## Supported Models

- LLaVA-1.5
- LLaVA-NeXT
- Qwen2.5-VL

The framework keeps the pretrained VLM backbone frozen and only performs inference-time visual token sparsification.

---

## Usage

### MME Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```

## Sparsification Configuration

The primary sparsification budget is controlled by:

```bash
--retained_tokens
```


## Main Components

### Instruction-Conditioned Token Pruning

- Predicts instruction category using a lightweight router.
- Selects task-relevant attention heads.
- Computes instruction-aware visual token relevance scores.
- Retains the most informative visual tokens under a given budget.

### Density-Aware Token Recycling

- Aggregates discarded tokens into compact representative tokens.
- Preserves complementary visual information.
- Mitigates information loss under aggressive sparsification.

---

## Benchmark Results

### LLaVA-1.5-7B (64 Tokens)

| Method | MME | POPE | TextVQA | SQA | GQA |
|----------|----------|----------|----------|----------|----------|
| FastV | 1263 | 48.0 | 51.1 | 51.1 | 46.1 |
| SparseVLM | 1591 | 77.5 | 53.4 | 69.8 | 53.8 |
| VisionZip | 1690 | 77.0 | 55.5 | 69.0 | 55.1 |
| PACT | 1572 | 78.8 | 53.6 | 68.2 | 54.1 |
| **ICVS** | **1711** | **79.7** | **53.9** | **70.5** | **54.3** |

More benchmark results can be found in our paper.

---


## Acknowledgment

This repository is built upon several excellent open-source projects:

- SparseVLMs
- LLaVA

We sincerely thank the authors for making their code publicly available.

---

## License

This project is released under the Apache 2.0 License.
