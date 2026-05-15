# ============================================
# 文件1: llava/classifier/prompt_classifier.py
# 位置: LLaVA/llava/classifier/prompt_classifier.py
# ============================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Tuple, Dict, Optional, List
import os
import time

# 14类任务类型映射
CATEGORY_MAPPING = {
    "existence": 0,
    "count": 1,
    "position": 2,
    "color": 3,
    "posters": 4,
    "celebrity": 5,
    "scene": 6,
    "landmark": 7,
    "artwork": 8,
    "OCR": 9,
    "commonsense_reasoning": 10,
    "numerical_calculation": 11,
    "text_translation": 12,
    "code_reasoning": 13
}

# ID到类别名称的反向映射
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_MAPPING.items()}


class PromptClassifier(nn.Module):
    """基于BERT的Prompt分类器 - 14类任务"""
    def __init__(self, model_name='/home/cwd/models/bert-tiny', num_classes=14, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout) #添加dropout层
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output) #应用dropout
        logits = self.classifier(pooled_output)
        return logits


class PromptTaskClassifier:
    """用于推理的分类器封装类"""
    def __init__(self, model_path: str, tokenizer_path: str = None, num_classes: int = 14):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = PromptClassifier(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.id_to_category = ID_TO_CATEGORY
        
        # 🔥 统计信息
        self.total_inference_time_ms = 0.0  # 总推理时间（毫秒）
        self.num_predictions = 0  # 预测次数
        self.total_flops = 0.0  # 总FLOPs
        
    def _estimate_flops(self, seq_length: int = 128) -> float:
        """估算分类器的FLOPs
        
        Args:
            seq_length: 序列长度（默认128）
        
        Returns:
            估算的FLOPs数量
        """
        # 获取BERT配置
        config = self.model.encoder.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size
        
        # BERT的FLOPs计算
        # 1. Embedding: vocab_size * hidden_size * seq_length (可以忽略，因为是查找表)
        # 2. 每层的Attention:
        #    - QKV projection: 3 * seq_length * hidden_size^2
        #    - Attention scores: seq_length^2 * hidden_size
        #    - Attention output: seq_length * hidden_size^2
        attention_flops_per_layer = (
            3 * seq_length * hidden_size * hidden_size +  # QKV
            seq_length * seq_length * hidden_size +  # Attention scores
            seq_length * hidden_size * hidden_size  # Output projection
        )
        
        # 3. 每层的FFN:
        #    - Up projection: seq_length * hidden_size * intermediate_size
        #    - Down projection: seq_length * intermediate_size * hidden_size
        ffn_flops_per_layer = (
            seq_length * hidden_size * intermediate_size +
            seq_length * intermediate_size * hidden_size
        )
        
        # 4. Layer Norm (可忽略，计算量很小)
        
        # 总FLOPs = num_layers * (attention + ffn)
        total_flops = num_layers * (attention_flops_per_layer + ffn_flops_per_layer)
        
        # 5. Pooler和Classifier head:
        #    - Pooler: hidden_size * hidden_size (可忽略，只取[CLS] token)
        #    - Classifier: hidden_size * num_classes
        classifier_flops = hidden_size * self.model.classifier.out_features
        
        total_flops += classifier_flops
        
        return float(total_flops)
        
    def predict(self, prompt: str, enable_timing: bool = False) -> Tuple[int, float, str, Optional[Dict]]:
        """预测prompt的任务类型
        
        Args:
            prompt: 输入文本
            enable_timing: 是否启用计时统计
        
        Returns:
            task_id: 任务ID (0-13)
            confidence: 置信度
            category: 任务类别名称
            stats: 统计信息字典（如果enable_timing=True）
        """
        stats = None
        
        if enable_timing:
            # 开始计时
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
        
        encoding = self.tokenizer(
            prompt,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 估算FLOPs
        seq_length = input_ids.shape[1]
        flops = self._estimate_flops(seq_length)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        if enable_timing:
            # 结束计时
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000.0
            
            # 更新统计信息
            self.total_inference_time_ms += inference_time_ms
            self.num_predictions += 1
            self.total_flops += flops
            
            stats = {
                'inference_time_ms': inference_time_ms,
                'flops': flops,
            }
        
        task_id = predicted.item()
        category = self.id_to_category.get(task_id, "Unknown")
        
        # 总是返回stats（如果未启用计时则为None），保持返回值数量一致
        if not enable_timing:
            stats = None
        return task_id, confidence.item(), category, stats
    
    def get_stats(self) -> Dict:
        """获取统计信息
        
        Returns:
            包含平均延迟、总延迟、平均FLOPs等统计信息的字典
        """
        if self.num_predictions == 0:
            return {
                'num_predictions': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0,
                'total_flops': 0.0,
                'avg_flops': 0.0,
                'total_flops_g': 0.0,  # GFLOPs
                'avg_flops_g': 0.0,
            }
        
        avg_time_ms = self.total_inference_time_ms / self.num_predictions
        avg_flops = self.total_flops / self.num_predictions
        
        return {
            'num_predictions': self.num_predictions,
            'total_time_ms': self.total_inference_time_ms,
            'avg_time_ms': avg_time_ms,
            'total_flops': self.total_flops,
            'avg_flops': avg_flops,
            'total_flops_g': self.total_flops / 1e9,  # GFLOPs
            'avg_flops_g': avg_flops / 1e9,
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_inference_time_ms = 0.0
        self.num_predictions = 0
        self.total_flops = 0.0


class PromptTaskClassifierV2(PromptTaskClassifier):
    """Threshold-based multi-task router using the original classifier weights.

    V2 keeps every task whose softmax confidence reaches min_confidence. It can
    optionally cap the selected task count, but it does not force exactly k
    outputs. If no task passes the threshold, it falls back to the top-1 task by
    default so generation still has a valid route.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        num_classes: int = 14,
        min_confidence: float = 0.20,
        max_tasks: Optional[int] = None,
        fallback_to_top1: bool = True,
    ):
        super().__init__(model_path, tokenizer_path=tokenizer_path, num_classes=num_classes)
        self.min_confidence = float(min_confidence)
        self.max_tasks = max_tasks
        self.fallback_to_top1 = fallback_to_top1

    def predict_multi(self, prompt: str, enable_timing: bool = False) -> Tuple[Dict[int, float], float, List[str], Optional[Dict]]:
        stats = None

        if enable_timing:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

        encoding = self.tokenizer(
            prompt,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        seq_length = input_ids.shape[1]
        flops = self._estimate_flops(seq_length)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0]

        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        selected = []
        for prob, task_id in zip(sorted_probs, sorted_ids):
            prob_value = float(prob.item())
            if prob_value < self.min_confidence:
                continue
            selected.append((int(task_id.item()), prob_value))
            if self.max_tasks is not None and len(selected) >= self.max_tasks:
                break

        if not selected and self.fallback_to_top1:
            selected = [(int(sorted_ids[0].item()), float(sorted_probs[0].item()))]

        task_confidences = {task_id: conf for task_id, conf in selected}
        categories = [self.id_to_category.get(task_id, "Unknown") for task_id in task_confidences]
        confidence = max(task_confidences.values()) if task_confidences else 0.0

        if enable_timing:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000.0

            self.total_inference_time_ms += inference_time_ms
            self.num_predictions += 1
            self.total_flops += flops

            stats = {
                'inference_time_ms': inference_time_ms,
                'flops': flops,
                'min_confidence': self.min_confidence,
                'max_tasks': self.max_tasks,
                'num_selected_tasks': len(task_confidences),
            }

        return task_confidences, confidence, categories, stats

    def predict(self, prompt: str, enable_timing: bool = False) -> Tuple[Dict[int, float], float, List[str], Optional[Dict]]:
        return self.predict_multi(prompt, enable_timing=enable_timing)
