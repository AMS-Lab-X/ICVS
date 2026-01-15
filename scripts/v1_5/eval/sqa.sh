#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /home/cwd/models/llava-v1.5-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/cwd/data/ScienceQA/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-64.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens 64 
    # --classifier-path  /home/cwd/codes/SparseVLMs/checkpoints/prompt_classifier_add/best_model.pth

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-64.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-64_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-64_result.json