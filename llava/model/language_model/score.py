import torch
import torch.nn as nn
import torch.nn.functional as F
import os

FIRST_PRUNING_LAYER = 2

layer_dict = {2:0, 6:1, 15:2}

sparse_token_list_192 = [300, 200, 110]
sparse_token_list_128 = [303, 110, 36]
sparse_token_list_64 = [66, 30, 17]
sparse_token_list_576 = [576, 576, 576]
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


# recycle_token_list_64 = [4, 3, 1]
recycle_token_list_64 = [5, 4, 1]
recycle_token_list_192= [2, 7, 3]
recycle_token_dict = {
    64: recycle_token_list_64,
    192:recycle_token_list_192,
}

def _parse_pruning_layers_env():
    raw_layers = os.environ.get("SPARSEVLM_PRUNING_LAYERS", "").strip()
    if not raw_layers:
        return None

    layers = []
    for item in raw_layers.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            layers.append(int(item))
        except ValueError:
            pass

    return layers or None


def _parse_int_list_env(name):
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return None

    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            pass

    return values or None


def get_pruning_stage_index(layer_idx):
    env_layers = _parse_pruning_layers_env()
    if env_layers is not None:
        try:
            return env_layers.index(int(layer_idx))
        except ValueError:
            return 0
    return layer_dict[layer_idx]


def resolve_sparse_token_budget(retained_tokens, layer_idx):
    env_layers = _parse_pruning_layers_env()
    if env_layers is not None and len(env_layers) == 1:
        return int(os.environ.get("SPARSEVLM_SINGLE_LAYER_TOKEN_BUDGET", retained_tokens))

    sparse_token_list = (
        _parse_int_list_env(f"SPARSEVLM_SPARSE_TOKEN_LIST_{retained_tokens}")
        or _parse_int_list_env("SPARSEVLM_SPARSE_TOKEN_LIST")
    )
    if sparse_token_list is not None:
        return sparse_token_list[get_pruning_stage_index(layer_idx)]

    sparse_token_list = sparse_token_dict[retained_tokens]
    return sparse_token_list[get_pruning_stage_index(layer_idx)]


def resolve_recycle_token_budget(retained_tokens, layer_idx):
    env_layers = _parse_pruning_layers_env()
    if env_layers is not None and len(env_layers) == 1:
        return int(os.environ.get("SPARSEVLM_SINGLE_LAYER_RECYCLE_TOKENS", 0))

    recycle_token_list = (
        _parse_int_list_env(f"SPARSEVLM_RECYCLE_TOKEN_LIST_{retained_tokens}")
        or _parse_int_list_env("SPARSEVLM_RECYCLE_TOKEN_LIST")
    )
    if recycle_token_list is not None:
        return recycle_token_list[get_pruning_stage_index(layer_idx)]

    recycle_token_list = recycle_token_dict.get(retained_tokens)
    if recycle_token_list is None:
        return 0
    return recycle_token_list[get_pruning_stage_index(layer_idx)]


TASK_LABELS = [
    "existence", "count", "position", "color", "posters", 
    "celebrity", "scene", "landmark", "artwork", "OCR", 
    "commonsense_reasoning", "numerical_calculation", 
    "text_translation", "code_reasoning"
]

label2id = {label: i for i, label in enumerate(TASK_LABELS)}
NUM_TASK = len(TASK_LABELS)


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

class TaskHeadRouter:
    def __init__(self, task_to_head_map, task_labels, first_layer=FIRST_PRUNING_LAYER):
        self.task_to_head_map = task_to_head_map
        self.task_labels = task_labels
        self.first_layer = first_layer

    def _head_override(self, num_heads):
        head_id = os.environ.get("SPARSEVLM_FIRST_LAYER_HEAD_ID")
        if head_id is None:
            head_id = os.environ.get("HEAD_ID")
        if head_id is None or head_id == "":
            return None

        try:
            head_id = int(head_id)
        except ValueError:
            return None

        if not 0 <= head_id < num_heads:
            return None

        return head_id

    def _iter_task_items(self, task_id):
        if task_id is None:
            return []
        if isinstance(task_id, dict):
            return task_id.items()
        if isinstance(task_id, (list, tuple, set)):
            return [(single_task_id, 1.0) for single_task_id in task_id]
        return [(task_id, 1.0)]

    def head_weights(self, task_id, num_heads):
        head_weights = {}
        for single_task_id, weight in self._iter_task_items(task_id):
            try:
                single_task_id = int(single_task_id)
                weight = float(weight)
            except (TypeError, ValueError):
                continue
            if weight <= 0 or single_task_id not in self.task_to_head_map:
                continue

            head_id = self.task_to_head_map[single_task_id]
            if not 0 <= head_id < num_heads:
                continue
            head_weights[head_id] = head_weights.get(head_id, 0.0) + weight

        return [(head_id, weight) for head_id, weight in head_weights.items()]

    def route_names(self, task_id):
        names = []
        for single_task_id, _ in self._iter_task_items(task_id):
            try:
                single_task_id = int(single_task_id)
            except (TypeError, ValueError):
                continue
            if 0 <= single_task_id < len(self.task_labels):
                names.append(self.task_labels[single_task_id])
        return names

    def select_attention(self, attention_weights, layer_idx, task_id=None):
        _, num_heads, _, _ = attention_weights.shape
        if layer_idx != self.first_layer:
            return attention_weights.mean(1)

        head_override = self._head_override(num_heads)
        if head_override is not None:
            return attention_weights[:, head_override]

        head_weights = self.head_weights(task_id, num_heads)
        if head_weights:
            head_ids = [head_id for head_id, _ in head_weights]
            weights = torch.tensor(
                [weight for _, weight in head_weights],
                dtype=attention_weights.dtype,
                device=attention_weights.device,
            )
            weights = weights / weights.sum().clamp_min(1e-6)
            return (attention_weights[:, head_ids] * weights.view(1, -1, 1, 1)).sum(1)

        return attention_weights.mean(1)


TASK_HEAD_ROUTER = TaskHeadRouter(TASK_TO_HEAD_MAP, TASK_LABELS)


def build_attention_topk_mask(self_attn_weights, v_token_start, v_token_num,
                              text_token_start, t_token_idx, layer_idx,
                              retained_tokens, task_id=None):
    B, H, L, _ = self_attn_weights.shape
    self_attn_weights = TASK_HEAD_ROUTER.select_attention(self_attn_weights, layer_idx, task_id)

        

    t_token_idx = t_token_idx[1] + text_token_start
    relation_vis_text = self_attn_weights[:, t_token_idx, v_token_start: v_token_start+v_token_num]
    relation_vis_text = relation_vis_text.mean(1)
    relation_vis = relation_vis_text
    s_flag = True

    

    sparse_token_num = resolve_sparse_token_budget(retained_tokens, layer_idx)

    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        _, indices = torch.topk(relation_vis, min(sparse_token_num, v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    
    return mask, s_flag, relation_vis_text


get_sparse_token_num = resolve_sparse_token_budget
get_recycle_token_num = resolve_recycle_token_budget
attn_postprocess_topk = build_attention_topk_mask
