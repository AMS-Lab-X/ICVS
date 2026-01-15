#!/bin/bash
set -euo pipefail

MODEL_PATH="/home/cwd/models/llava-v1.5-7b"
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="/home/cwd/data/TextVQA/train_images"
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"

OUTPUT_DIR="./textvqa_eval"
LOG_DIR="${OUTPUT_DIR}/logs"
TXT_DIR="${OUTPUT_DIR}/textvqa_txt"
ANS_DIR="${OUTPUT_DIR}/tmp_answers"

mkdir -p "$LOG_DIR" "$TXT_DIR" "$ANS_DIR"

SUMMARY_CSV="${OUTPUT_DIR}/textvqa_summary.csv"
echo "head,accuracy" > "$SUMMARY_CSV"

for HEAD in {0..31}; do
    echo "========================================="
    echo "Running TextVQA with HEAD_ID = $HEAD"
    echo "========================================="

    export HEAD_ID=$HEAD

    ANSWER_FILE="${ANS_DIR}/head_${HEAD}.jsonl"
    LOG_FILE="${LOG_DIR}/head_${HEAD}.log"
    TXT_FILE="${TXT_DIR}/head_${HEAD}_textvqa.txt"

    rm -f "$ANSWER_FILE" "$LOG_FILE" "$TXT_FILE"

    ##############################################
    # 1) TextVQA 推理 —— 逻辑完全照抄 SQA.sh 的 Step1
    ##############################################
    echo "[Step 1] model_vqa_loader (HEAD=$HEAD)" | tee -a "$LOG_FILE"
    python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWER_FILE" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --retained_tokens 64 \
        >> "$LOG_FILE" 2>&1

    ##############################################
    # 2) TextVQA 评估 —— 完全照 SQA 的 Step2
    ##############################################
    echo -e "\n\n[Step 2] eval_textvqa (HEAD=$HEAD)" >> "$LOG_FILE"
    python -m llava.eval.eval_textvqa \
        --annotation-file "$ANNOTATION_FILE" \
        --result-file "$ANSWER_FILE" \
        >> "$LOG_FILE" 2>&1 || true

    ##############################################
    # 3) 提取 Accuracy —— 与 SQA 完全一致的抽取逻辑
    ##############################################
    awk '
        BEGIN{flag=0}
        /Accuracy/ && flag==0 { flag=1 }
        flag { print }
        /^=+$/ && flag==1 { exit }
    ' "$LOG_FILE" > "$TXT_FILE" || true

    # fallback（与 SQA 完全一致）
    if ! grep -q "Accuracy" "$TXT_FILE" 2>/dev/null; then
        echo "[WARN] primary block extraction failed for head $HEAD, fallback to grep lines" >> "$LOG_FILE"
        grep -E "Accuracy|Correct|accuracy" "$LOG_FILE" > "$TXT_FILE" || true
    fi

    echo "Saved extracted metrics -> $TXT_FILE"

    ##############################################
    # 4) 提取一个 Accuracy 数值，并写入 summary
    ##############################################
    ACC_LINE=$(grep -m1 -E "Accuracy[: ]" "$TXT_FILE" || true)
    if [[ -n "$ACC_LINE" ]]; then
        ACC_VAL=$(echo "$ACC_LINE" | grep -oE "[0-9]+(\.[0-9]+)?" | head -n1)
    else
        ACC_VAL=""
    fi

    echo "${HEAD},${ACC_VAL}" >> "$SUMMARY_CSV"

    ##############################################
    # 5) 删除 json（完全照 SQA）
    ##############################################
    rm -f "$ANSWER_FILE"

    echo "Head $HEAD done. Log -> $LOG_FILE ; Metrics -> $TXT_FILE"
done

echo "========================================="
echo "All HEAD_ID evaluations for TextVQA DONE!"
echo "Results/logs in: $OUTPUT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "========================================="
