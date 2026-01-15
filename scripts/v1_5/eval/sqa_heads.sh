#!/bin/bash
set -euo pipefail

MODEL_PATH="/home/cwd/models/llava-v1.5-7b"
QUESTION_FILE="./playground/data/eval/scienceqa/llava_test_CQM-A.json"
IMAGE_FOLDER="/home/cwd/data/ScienceQA/test"
BASE_DIR="./playground/data/eval/scienceqa"

OUTPUT_DIR="./scienceqa_eval"
LOG_DIR="${OUTPUT_DIR}/logs"
TXT_DIR="${OUTPUT_DIR}/sqa_txt"
ANS_DIR="${OUTPUT_DIR}/tmp_answers"

mkdir -p "$LOG_DIR" "$TXT_DIR" "$ANS_DIR"

# CSV summary (可选)
SUMMARY_CSV="${OUTPUT_DIR}/scienceqa_summary.csv"
echo "head,accuracy" > "$SUMMARY_CSV"

for HEAD in {0..31}; do
    echo "========================================="
    echo "Running ScienceQA with HEAD_ID = $HEAD"
    echo "========================================="

    export HEAD_ID=$HEAD

    ANSWER_FILE="${ANS_DIR}/head_${HEAD}.jsonl"        # 临时答案文件，运行结束后会删除
    OUTPUT_JSON="${ANS_DIR}/head_${HEAD}_output.jsonl" # 如果 eval 会生成输出文件，放在临时目录
    RESULT_JSON="${ANS_DIR}/head_${HEAD}_result.json"  # 同上（视 eval 脚本行为）
    LOG_FILE="${LOG_DIR}/head_${HEAD}.log"
    SQA_TXT="${TXT_DIR}/head_${HEAD}_sqa.txt"

    # 清理旧文件
    rm -f "$ANSWER_FILE" "$OUTPUT_JSON" "$RESULT_JSON" "$LOG_FILE" "$SQA_TXT"

    ##############################################
    # 1) 运行推理（model_vqa_science），把所有 stdout/stderr 写入 log
    ##############################################
    echo "[Step 1] model_vqa_science (HEAD=$HEAD)" | tee -a "$LOG_FILE"
    python -m llava.eval.model_vqa_science \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$ANSWER_FILE" \
        --single-pred-prompt \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --retained_tokens 64 \
        >> "$LOG_FILE" 2>&1

    ##############################################
    # 2) 运行评估脚本 eval_science_qa.py（全部输出也写入 log）
    ##############################################
    echo -e "\n\n[Step 2] eval_science_qa.py (HEAD=$HEAD)" >> "$LOG_FILE"
    python llava/eval/eval_science_qa.py \
        --base-dir "$BASE_DIR" \
        --result-file "$ANSWER_FILE" \
        --output-file "$OUTPUT_JSON" \
        --output-result "$RESULT_JSON" \
        >> "$LOG_FILE" 2>&1 || true

    ##############################################
    # 3) 从 log 中提取关键指标块并保存到单独 txt
    #    （从首个 "Accuracy" 行开始，直到遇到分隔线 ====== 或文件结束）
    ##############################################
    awk '
        BEGIN{flag=0}
        /Accuracy/ && flag==0 { flag=1 }
        flag { print }
        /^=+$/ && flag==1 { exit }
    ' "$LOG_FILE" > "$SQA_TXT" || true

    # 备用：如果上面没有抓到（格式差异），则抓取包含 Accuracy/F1 的所有行
    if ! grep -q "Accuracy" "$SQA_TXT" 2>/dev/null; then
        echo "[WARN] primary block extraction failed for head $HEAD, fallback to grep lines" >> "$LOG_FILE"
        grep -E "Accuracy|F1|accuracy|Overall|Per-type|Correct" "$LOG_FILE" > "$SQA_TXT" || true
    fi

    echo "Saved extracted metrics -> $SQA_TXT"

    ##############################################
    # 4) 可选：从提取的文本中获取单值 Accuracy 并附到 summary CSV
    ##############################################
    # 尝试解析首个出现的 Accuracy: value
    ACC_LINE=$(grep -m1 -E "Accuracy[: ]" "$SQA_TXT" || true)
    if [[ -n "$ACC_LINE" ]]; then
        # 提取数字（浮点）
        ACC_VAL=$(echo "$ACC_LINE" | grep -oE "[0-9]+(\.[0-9]+)?" | head -n1)
    else
        ACC_VAL=""
    fi
    echo "${HEAD},${ACC_VAL}" >> "$SUMMARY_CSV"

    ##############################################
    # 5) 清理临时 json（你要求不保存 json）
    ##############################################
    rm -f "$ANSWER_FILE" "$OUTPUT_JSON" "$RESULT_JSON"

    echo "Head $HEAD done. Log -> $LOG_FILE ; Metrics -> $SQA_TXT"
done

echo "========================================="
echo "All HEAD_ID evaluations for ScienceQA DONE!"
echo "Results/logs in: $OUTPUT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "========================================="
