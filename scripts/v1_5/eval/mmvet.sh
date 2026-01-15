#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /home/cwd/models/llava-v1.5-7b \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-64.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens 64 \
    --classifier-path   /home/cwd/codes/SparseVLMs/checkpoints/prompt_classifier_add/best_model.pth   

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-64.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b-64.json