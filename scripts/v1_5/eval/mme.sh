#!/bin/bash
#可选
    # --visualize-pruning \
    # --visualization-output-dir ./visualizations \

export CUDA_VISIBLE_DEVICES=0

python -m llava.eval.model_vqa_loader \
    --model-path /home/cwd/models/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /home/cwd/codes/SparseVLMs-1.5/outputs/TADHS/MME/head_classifier/llava-v1.5-7b-64.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens 64 \
    --log-file  /home/cwd/codes/SparseVLMs-1.5/outputs/TADHS/MME/head_classifier/llava-v1.5-7b-64.log \
    --classifier-path ./checkpoints/prompt_classifier_add/best_model.pth

cd ./playground/data/eval/MME
mkdir -p answers
ln -sf /home/cwd/codes/SparseVLMs-1.5/outputs/TADHS/MME/head_classifier/llava-v1.5-7b-64.jsonl \
      answers/llava-v1.5-7b-64.jsonl

python convert_answer_to_mme.py --experiment llava-v1.5-7b-64

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-64

