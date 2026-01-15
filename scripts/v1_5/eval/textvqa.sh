# #!/bin/bash
    # --visualize-pruning \
    # --visualization-output-dir ./visualizations_textvqa_64_TADHS \

# python -m llava.eval.model_vqa_loader \
#     --model-path /home/cwd/models/llava-v1.5-7b \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /home/cwd/data/TextVQA/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-64.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --retained_tokens 64 \
#     --classifier-path  /home/cwd/codes/SparseVLMs/checkpoints/prompt_classifier_add/best_model.pth

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file /home/cwd/codes/SparseVLMs/playground/data/eval/textvqa/answers/llava-v1.5-7b-64.jsonl

#!/bin/bash

#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /home/cwd/models/llava-v1.5-7b \
        --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder /home/cwd/data/TextVQA/train_images \
        --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${CHUNKS}_${IDX}_64.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --retained_tokens 64 \
        --classifier-path /home/cwd/codes/SparseVLMs/checkpoints/prompt_classifier_add/best_model.pth &
done

wait

output_file=./playground/data/eval/textvqa/answers/${CKPT}/merge_64.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/textvqa/answers/${CKPT}/${CHUNKS}_${IDX}_64.jsonl >> "$output_file"
done
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file

