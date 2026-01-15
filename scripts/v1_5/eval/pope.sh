#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa_loader \
    --model-path /home/cwd/models/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/cwd/data/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-64.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens 64 

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-64.jsonl