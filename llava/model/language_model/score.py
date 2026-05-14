# ============================================
# 文件1: llava/model/language_model/score.py (修改版)
# 关键修改：支持任务自适应的头选择 + FFT 频谱分析
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

layer_dict = {2:0, 6:1, 15:2}

sparse_token_list_192 = [300, 200, 110]
sparse_token_list_128 = [303, 110, 36]
sparse_token_list_64 = [66, 30, 17]
sparse_token_list_576 = [576, 576, 576]
# sparse_token_list_576 = [1, 1, 1]
sparse_token_list_48 = [30, 10, 7]
sparse_token_list_96 = [120, 64, 48]
sparse_token_list_288 = [288, 288, 288]

sparse_token_dict = {
    192: sparse_token_list_192,
    128: sparse_token_list_128,
    64: sparse_token_list_64,
    576: sparse_token_list_576,
    48: sparse_token_list_48,
    96: sparse_token_list_96,
    288: sparse_token_list_288
}

TASK_LABELS = [
    "existence", "count", "position", "color", "posters", 
    "celebrity", "scene", "landmark", "artwork", "OCR", 
    "commonsense_reasoning", "numerical_calculation", 
    "text_translation", "code_reasoning"
]

label2id = {label: i for i, label in enumerate(TASK_LABELS)}
NUM_TASK = len(TASK_LABELS)

# # 🔥 任务到最优头的映射
# TASK_TO_HEAD_MAP = {
#     label2id["existence"]: 23,  #23 18 24 26 3 9 13 28 
#     label2id["count"]: 23, 
#     label2id["position"]: 22,
#     label2id["color"]: 23,    #26 3 9 22 23
#     label2id["posters"]: 2,
#     label2id["celebrity"]: 12,
#     label2id["scene"]: 31,
#     label2id["landmark"]: 5,
#     label2id["artwork"]: 5,
#     label2id["OCR"]: 28,
#     label2id["commonsense_reasoning"]: 19,
#     label2id["numerical_calculation"]: 5,
#     label2id["text_translation"]: 18,
#     label2id["code_reasoning"]: 25,
# }

# 🔥 任务到最优头的映射
TASK_TO_HEAD_MAP = {
    label2id["existence"]: 3, 
    label2id["count"]: 8, 
    label2id["position"]: 22,
    label2id["color"]: 23,    #26 3 9 22 23
    label2id["posters"]: 2,
    label2id["celebrity"]: 12,
    label2id["scene"]: 31,
    label2id["landmark"]: 0,
    label2id["artwork"]: 7,
    label2id["OCR"]: 1,
    label2id["commonsense_reasoning"]: 20,
    label2id["numerical_calculation"]: 28,
    label2id["text_translation"]: 31,
    label2id["code_reasoning"]: 22,
}

# # 🔥 默认头（如果任务ID不在映射中）
# DEFAULT_HEAD = 26




def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, 
                          text_token_start, t_token_idx, layer_idx, 
                          retained_tokens, task_id=None):
    '''
    self_attn_weights: [B, H, L, L]
    task_id: 🔥 新增参数 - 任务类别ID, 用于选择最优注意力头
    '''
    B, H, L, _ = self_attn_weights.shape  # H = 32 (注意力头数量)
    
    # 🔥 原有的注意力头选择逻辑
    if layer_idx == 2:
        # 🔥 核心修改：根据任务ID动态选择头
        if task_id is not None and task_id in TASK_TO_HEAD_MAP:
            HEAD_ID = TASK_TO_HEAD_MAP[task_id]
            print(f"[Score] Using task-specific head: Task={TASK_LABELS[task_id]}, Head={HEAD_ID}")
            self_attn_weights = self_attn_weights[:, HEAD_ID]
        else:
            # 如果没有提供task_id或task_id不在映射中，使用平均注意力
            if task_id is not None:
                print(f"[Score] Task ID {task_id} not in map, using average head")
            self_attn_weights = self_attn_weights.mean(1)
    else:
        # 其他层使用平均注意力
        self_attn_weights = self_attn_weights.mean(1)

        

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num] #选取文本Q-视觉K
    relation_vis_text = relation_vis_text.mean(1)
    relation_vis = relation_vis_text
    s_flag = True #决定是否回收

    

    sparse_token_list = sparse_token_dict[retained_tokens]
    if layer_idx == 2:
        print(f"[Score] retained_tokens={retained_tokens}, sparse_token_list={sparse_token_list}")


    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(sparse_token_list[layer_dict[layer_idx]], v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    
    return mask, s_flag, relation_vis_text