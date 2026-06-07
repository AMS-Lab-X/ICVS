# Instruction-Conditioned Visual Token Sparsification (ICVS)

Efficient inference for vision-language models (VLMs) via **instruction-conditioned visual token sparsification**. This repository implements the ICVS framework proposed in the paper:

> **Instruction-conditioned visual token sparsification for efficient vision-language model inference**  
> Weidong Cao, Xi Zhang, Chengyang Li, Danjun Liu, Yongqiang Xie, Zhongbo Li  
> [PDF link](./cas-dc-sample(3).pdf)  

---

## Highlights

- Reformulates visual token pruning as an **instruction-aware process**.  
- Leverages **task-relevant attention heads** to guide token selection.  
- Introduces **density-aware token recycling** to preserve complementary visual information.  
- Achieves **better accuracy–efficiency trade-off** for high-resolution multimodal tasks.  

---

## Installation

1. **Clone the repository** and navigate to the SparseVLMs folder:

```bash
git clone https://github.com/AMS-Lab-X/ICVS.git
cd SparseVLMs
