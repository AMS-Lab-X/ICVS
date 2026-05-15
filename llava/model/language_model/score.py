
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




def _get_first_pruning_layer_head_override(num_heads):
    head_id = os.environ.get("SPARSEVLM_FIRST_LAYER_HEAD_ID")
    if head_id is None:
        head_id = os.environ.get("HEAD_ID")
    if head_id is None or head_id == "":
        return None

    try:
        head_id = int(head_id)
    except ValueError:
        print(f"[Score] Ignore invalid head override: {head_id}")
        return None

    if not 0 <= head_id < num_heads:
        print(f"[Score] Ignore out-of-range head override: {head_id}, num_heads={num_heads}")
        return None

    return head_id


def _get_task_head_weights(task_id, num_heads):
    if task_id is None:
        return []

    if isinstance(task_id, dict):
        task_items = task_id.items()
    elif isinstance(task_id, (list, tuple, set)):
        task_items = [(single_task_id, 1.0) for single_task_id in task_id]
    else:
        task_items = [(task_id, 1.0)]

    head_weights = {}
    for single_task_id, weight in task_items:
        try:
            single_task_id = int(single_task_id)
            weight = float(weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0 or single_task_id not in TASK_TO_HEAD_MAP:
            continue

        head_id = TASK_TO_HEAD_MAP[single_task_id]
        if not 0 <= head_id < num_heads:
            continue
        head_weights[head_id] = head_weights.get(head_id, 0.0) + weight

    return [(head_id, weight) for head_id, weight in head_weights.items()]


def _format_task_route(task_id):
    if isinstance(task_id, dict):
        task_ids = task_id.keys()
    elif isinstance(task_id, (list, tuple, set)):
        task_ids = task_id
    else:
        task_ids = [task_id]

    names = []
    for single_task_id in task_ids:
        try:
            single_task_id = int(single_task_id)
        except (TypeError, ValueError):
            continue
        if 0 <= single_task_id < len(TASK_LABELS):
            names.append(TASK_LABELS[single_task_id])
    return names


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
        head_override = _get_first_pruning_layer_head_override(H)
        if head_override is not None:
            print(f"[Score] Using fixed first-pruning-layer head: Head={head_override}")
            self_attn_weights = self_attn_weights[:, head_override]
        elif _get_task_head_weights(task_id, H):
            head_weights = _get_task_head_weights(task_id, H)
            head_ids = [head_id for head_id, _ in head_weights]
            weights = torch.tensor(
                [weight for _, weight in head_weights],
                dtype=self_attn_weights.dtype,
                device=self_attn_weights.device,
            )
            weights = weights / weights.sum().clamp_min(1e-6)
            print(f"[Score] Using task-specific heads: Tasks={_format_task_route(task_id)}, Heads={head_ids}")
            self_attn_weights = (self_attn_weights[:, head_ids] * weights.view(1, -1, 1, 1)).sum(1)
            # self_attn_weights = self_attn_weights.mean(1)
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
