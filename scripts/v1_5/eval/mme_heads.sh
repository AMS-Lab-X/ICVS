#!/bin/bash

LOG_DIR="/home/cwd/codes/SparseVLMs/logs/mme_heads"
mkdir -p $LOG_DIR

SCORES_FILE="scores.txt"
rm -f $SCORES_FILE

HEAD_SCORE_DIR="head_scores"
mkdir -p $HEAD_SCORE_DIR

for HEAD in {0..31}
do
    echo "========================================="
    echo "Running HEAD_ID = $HEAD"
    echo "========================================="

    export HEAD_ID=$HEAD

    # 运行你的 MME 测试
    bash /home/cwd/codes/SparseVLMs/scripts/v1_5/eval/mme.sh > ${LOG_DIR}/head_${HEAD}.log 2>&1

    LOG_FILE="${LOG_DIR}/head_${HEAD}.log"
    OUT_FILE="${HEAD_SCORE_DIR}/head_${HEAD}.txt"

    #############################################################
    # 提取 total score（Perception 的）
    #############################################################
    PERCEPTION_TOTAL=$(grep "Perception ===========" -n $LOG_FILE | while read -r line ; do 
        start=$(echo "$line" | cut -d: -f1)
        sed -n "$((start+1))p" $LOG_FILE | grep "total score" | awk '{print $NF}'
        break
    done)

    # 记入 summary scores.txt
    echo "$HEAD $PERCEPTION_TOTAL" >> $SCORES_FILE

    #############################################################
    # ★★★ 保存完整的 Perception + Cognition 内容 ★★★
    #############################################################

    # 找 Perception 块起始行号
    START_LINE=$(grep -n "=========== Perception ===========" $LOG_FILE | awk -F: '{print $1}')
    # 找 Cognition 块结束（直到下一段空行结束）
    END_LINE=$(grep -n "code_reasoning" $LOG_FILE | awk -F: '{print $1}')

    # 如果找不到（理论上不会发生）
    if [[ -z "$START_LINE" || -z "$END_LINE" ]]; then
        echo "Warning: Cannot locate output block for HEAD $HEAD"
        cp $LOG_FILE $OUT_FILE
        continue
    fi

    # ±5 行保险范围，完整包含空行 & 分隔符
    START_LINE=$((START_LINE))
    END_LINE=$((END_LINE + 2))

    echo "Head $HEAD Full Evaluation Output" > $OUT_FILE
    echo "==========================================" >> $OUT_FILE

    sed -n "${START_LINE},${END_LINE}p" $LOG_FILE >> $OUT_FILE

    echo "Saved full block to $OUT_FILE"
done

echo "All heads finished"
echo "Summary in scores.txt"
echo "Full evaluation blocks saved in head_scores/"
