#!/bin/bash
#可选
    # --visualize-pruning \
    # --visualization-output-dir ./visualizations \

export CUDA_VISIBLE_DEVICES=0

# python -m llava.eval.model_vqa_loader \
#     --model-path /home/cwd/models/llava-v1.5-7b \
#     --question-file ./playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b-64.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --retained_tokens 64 \
#     --classifier-path ./checkpoints/prompt_classifier_add/best_model.pth

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-64

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-64

