# Instruction-conditioned Visual Token Sparsification for Efficient Vision-Language Model Inference

## Highlights

- Reformulates visual token pruning as an instruction-aware process.
- Leverages task-relevant attention heads to guide token selection.
- Introduces density-aware token recycling to preserve complementary visual information.
- Achieves better accuracy-efficiency trade-off for high-resolution multimodal tasks.

## Installation

1. Clone this repository and navigate to SparseVLMs folder
```bash
git clone https://github.com/AMS-Lab-X/ICVS.git
cd SparseVLMs
```

2. Install necessary package
```Shell
conda create -n ICVS python=3.10 -y
conda activate ICVS
pip install -e .
pip install transformers==4.37.2
pip install flash_attn==2.3.3
```
3. Download Multimodal Benchmark

Please follow the detailed instruction in [LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).
```

## Supported Models

- LLaVA-1.5
- LLaVA-NeXT
- Qwen2.5-VL

The framework keeps the pretrained VLM backbone frozen and only applies inference-time sparsification.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Example for MME evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```

The main sparsification budget is controlled by `--retained_tokens`. Layer-wise token budgets and recycling behavior can be adjusted in `llava/model/language_model/score.py`.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgment

This repository is developed based on the open-source efforts of [SparseVLMs](https://github.com/Gumpest/SparseVLMs), [LLaVA](https://github.com/haotian-liu/LLaVA), [MiniGemini](https://github.com/dvlab-research/MGM), and [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA).
