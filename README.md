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

## Method Overview

ICVS consists of two main components.

### 1. Instruction-Conditioned Token Pruning

The instruction-conditioned routing is implemented as a two-stage procedure.
First, a lightweight prompt classifier predicts the task category from the input instruction. The implementation is provided in:

```text
llava/classifier/prompt_classifier.py
```
During evaluation, the classifier is invoked in:

```text
llava/eval/model_vqa_loader.py
```
The predicted task id is passed to the VLM generation function:

```python
model.generate(..., task_id=predicted_task_id)
```
Second, the predicted task category is used to retrieve the corresponding task-aligned attention head from an offline task-to-head lookup table. The lookup table is implemented in:
```text
llava/model/language_model/score.py
```
Specifically, the offline lookup table is stored as:

```python
TASK_TO_HEAD_MAP
```
The hard-coded dictionary is only the storage form of the offline task-to-head lookup table for efficient access during inference. It is not a replacement for the entire instruction-conditioned routing process.

### 2. Density-Aware Token Recycling
After sparse token selection, ICVS further aggregates part of the discarded visual tokens into compact representative tokens. This density-aware token recycling strategy helps preserve complementary visual information and mitigates information loss under aggressive sparsification.

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
Please make sure that PyTorch, CUDA, and FlashAttention are installed according to your local GPU and CUDA environment.

## Model and Checkpoint Preparation

### 1. VLM Backbone

Please prepare the corresponding pretrained VLM backbone, such as LLaVA-1.5-7B, following the official LLaVA instructions:

```text
https://github.com/haotian-liu/LLaVA
```

You may specify the model path in the evaluation script, for example:

```bash
--model-path /path/to/llava-v1.5-7b
```

### 2. Prompt Classifier Checkpoint

The prompt classifier checkpoint can be downloaded from:

```text
https://drive.google.com/file/d/1qTWM9WM4rEnqWUgBbA5B61x-JJboQ7mX/view?usp=drive_link
```
After downloading, place the checkpoint at:

```text
./checkpoints/prompt_classifier/best_model.pth
```
or specify its path manually using:

```bash
--classifier-path /path/to/best_model.pth
```

The classifier path used in the example script is:

```bash
--classifier-path ./checkpoints/prompt_classifier/best_model.pth
```

---

## Usage
### MME Evaluation
An example evaluation script is provided at:

```text
scripts/v1_5/eval/mme.sh
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh
```

The main sparsification budget is controlled by:

```bash
--retained_tokens
```

For example:

```bash
--retained_tokens 64
```

To enable instruction-conditioned routing, provide the prompt classifier checkpoint:

```bash
--classifier-path ./checkpoints/prompt_classifier/best_model.pth
```
A typical evaluation command contains the following key arguments:

```bash
python -m llava.eval.model_vqa_loader \
    --model-path /path/to/llava-v1.5-7b \
    --question-file /path/to/questions.jsonl \
    --image-folder /path/to/images \
    --answers-file /path/to/answers.jsonl \
    --retained_tokens 64 \
    --classifier-path ./checkpoints/prompt_classifier_add/best_model.pth
```

---

## Key Arguments

| Argument | Description |
|---|---|
| `--retained_tokens` | Number of visual tokens retained after sparsification. |
| `--classifier-path` | Path to the trained prompt classifier checkpoint. If omitted, the model falls back to the default attention aggregation strategy. |
| `--classifier-version` | Version of the prompt classifier. |
| `--classifier-min-confidence` | Minimum confidence threshold for classifier prediction. |
| `--classifier-max-tasks` | Maximum number of task categories used for routing. |

---

### 3. Download evaluation benchmarks

Please follow the official LLaVA evaluation instructions:

[LLaVA-Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

---

## Implementation Details

### Prompt Classifier

The lightweight prompt classifier is implemented in:

```text
llava/classifier/prompt_classifier.py
```

It predicts the task category from the input instruction. The task category is represented as a task id and passed to the sparse VLM during generation.

### Task-to-Head Lookup Table

The task-to-head lookup table is implemented in:

```text
llava/model/language_model/score.py
```

The mapping is stored as:

```python
TASK_TO_HEAD_MAP
```

This lookup table is constructed offline by evaluating attention heads under a controlled pruning setting and recording the best-performing head for each predefined task category.

For efficient inference, the offline lookup table is stored as a hard-coded dictionary. This design avoids additional routing overhead during inference while keeping the routing process instruction-conditioned through the prompt classifier.

### Sparse Token Scoring

During inference, the selected task-aligned attention head is used to compute visual token relevance scores. The most informative visual tokens are retained according to the specified token budget.

The sparse token scoring and recycling functions are mainly implemented in:

```text
llava/model/language_model/score.py
```

### Density-Aware Token Recycling

Discarded visual tokens are not simply removed. ICVS aggregates part of them into compact representative tokens to preserve complementary visual information, especially under small token budgets.

---

## Supported Models

- LLaVA-1.5
- LLaVA-NeXT
- Qwen2.5-VL

The framework keeps the pretrained VLM backbone frozen and only performs inference-time visual token sparsification.

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
