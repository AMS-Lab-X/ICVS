#!/bin/bash

MODEL_PATH="/home/cwd/models/llava-v1.5-7b"
QUESTION_FILE="./playground/data/eval/pope/llava_pope_test.jsonl"
IMAGE_FOLDER="/home/cwd/data/pope/val2014"
ANNOTATION_DIR="./playground/data/eval/pope/coco"
OUTPUT_DIR="/home/cwd/codes/SparseVLMs/logs"
mkdir -p $OUTPUT_DIR/pope_heads
mkdir -p $OUTPUT_DIR/pope_scores

for HEAD in {0..31}
do
    echo "========================================="
    echo "Running HEAD_ID = $HEAD"
    echo "========================================="

    export HEAD_ID=$HEAD
    LOG_FILE="${OUTPUT_DIR}/pope_heads/head_${HEAD}.log"
    POPE_TXT="${OUTPUT_DIR}/pope_scores/head_${HEAD}.txt"

    ##############################################
    # 1. 运行 model_vqa_loader（保持原样输出到 log）
    ##############################################
    echo "[Step1] model_vqa_loader output" >> $LOG_FILE
    python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file ${OUTPUT_DIR}/answers_head_${HEAD}.jsonl \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --retained_tokens 64 \
        >> $LOG_FILE 2>&1

    ##############################################
    # 2. 运行 eval_pope.py（输出到 log + 单独保存）
    ##############################################
    echo -e "\n\n[Step2] eval_pope.py output" >> $LOG_FILE

    python llava/eval/eval_pope.py \
        --annotation-dir $ANNOTATION_DIR \
        --question-file $QUESTION_FILE \
        --result-file ${OUTPUT_DIR}/answers_head_${HEAD}.jsonl \
        >> $LOG_FILE 2>&1

    ##############################################
    # 3. ★★★ 提取 eval_pope 的特定部分保存到 txt ★★★
    ##############################################
    # 读取 eval_pope 输出（从 log 中提取）
    awk '
        /Category: adversarial/ {flag=1}
        flag {print}
        /Yes ratio:/ && NR>1 {nextline=1}
        nextline && /====================================/ {nextline=0; print; next}
        /final/ {flag=0}
    ' $LOG_FILE > $POPE_TXT

    echo "Saved pope metrics -> $POPE_TXT"

done

echo "All heads done!"
