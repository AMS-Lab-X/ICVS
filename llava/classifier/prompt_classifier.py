import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


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
    "code_reasoning": 13,
}

ID_TO_CATEGORY = {v: k for k, v in CATEGORY_MAPPING.items()}


class PromptClassifier(nn.Module):
    def __init__(self, model_name="/home/cwd/models/bert-tiny", num_classes=14, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class PromptTaskClassifier:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        num_classes: int = 14,
        encoder_model_name: str = "/home/cwd/models/bert-tiny",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if tokenizer_path is None:
            tokenizer_path = os.path.dirname(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = PromptClassifier(encoder_model_name, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.id_to_category = ID_TO_CATEGORY
        self.total_inference_time_ms = 0.0
        self.num_predictions = 0
        self.total_flops = 0.0

    def _estimate_flops(self, seq_length: int = 128) -> float:
        config = self.model.encoder.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        intermediate_size = config.intermediate_size

        attention_flops_per_layer = (
            3 * seq_length * hidden_size * hidden_size
            + seq_length * seq_length * hidden_size
            + seq_length * hidden_size * hidden_size
        )
        ffn_flops_per_layer = (
            seq_length * hidden_size * intermediate_size
            + seq_length * intermediate_size * hidden_size
        )
        classifier_flops = hidden_size * self.model.classifier.out_features
        return float(num_layers * (attention_flops_per_layer + ffn_flops_per_layer) + classifier_flops)

    def _encode(self, prompt: str):
        encoding = self.tokenizer(
            prompt,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def _record_stats(self, inference_time_ms: float, flops: float):
        self.total_inference_time_ms += inference_time_ms
        self.num_predictions += 1
        self.total_flops += flops

    def predict(self, prompt: str, enable_timing: bool = False) -> Tuple[int, float, str, Optional[Dict]]:
        stats = None

        if enable_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time() if enable_timing else None

        input_ids, attention_mask = self._encode(prompt)
        flops = self._estimate_flops(input_ids.shape[1])

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, 1)
            top_probs, top_ids = torch.topk(probs[0], k=min(2, probs.shape[1]))

        if enable_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_time_ms = (time.time() - start_time) * 1000.0
            self._record_stats(inference_time_ms, flops)
            stats = {
                "inference_time_ms": inference_time_ms,
                "flops": flops,
                "top_candidates": [
                    {
                        "task_id": int(task_id.item()),
                        "category": self.id_to_category.get(int(task_id.item()), "Unknown"),
                        "confidence": float(prob.item()),
                    }
                    for prob, task_id in zip(top_probs, top_ids)
                ],
            }

        task_id = int(predicted.item())
        category = self.id_to_category.get(task_id, "Unknown")
        return task_id, float(confidence.item()), category, stats

    def get_stats(self) -> Dict:
        if self.num_predictions == 0:
            return {
                "num_predictions": 0,
                "total_time_ms": 0.0,
                "avg_time_ms": 0.0,
                "total_flops": 0.0,
                "avg_flops": 0.0,
                "total_flops_g": 0.0,
                "avg_flops_g": 0.0,
            }

        avg_time_ms = self.total_inference_time_ms / self.num_predictions
        avg_flops = self.total_flops / self.num_predictions
        return {
            "num_predictions": self.num_predictions,
            "total_time_ms": self.total_inference_time_ms,
            "avg_time_ms": avg_time_ms,
            "total_flops": self.total_flops,
            "avg_flops": avg_flops,
            "total_flops_g": self.total_flops / 1e9,
            "avg_flops_g": avg_flops / 1e9,
        }

    def reset_stats(self):
        self.total_inference_time_ms = 0.0
        self.num_predictions = 0
        self.total_flops = 0.0


class PromptTaskClassifierV2(PromptTaskClassifier):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        num_classes: int = 14,
        min_confidence: float = 0.20,
        max_tasks: Optional[int] = None,
        fallback_to_top1: bool = True,
        encoder_model_name: str = "/home/cwd/models/bert-tiny",
    ):
        super().__init__(
            model_path,
            tokenizer_path=tokenizer_path,
            num_classes=num_classes,
            encoder_model_name=encoder_model_name,
        )
        self.min_confidence = float(min_confidence)
        self.max_tasks = max_tasks
        self.fallback_to_top1 = fallback_to_top1

    def predict_multi(
        self, prompt: str, enable_timing: bool = False
    ) -> Tuple[Dict[int, float], float, List[str], Optional[Dict]]:
        stats = None

        if enable_timing and self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time() if enable_timing else None

        input_ids, attention_mask = self._encode(prompt)
        flops = self._estimate_flops(input_ids.shape[1])

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

        fallback_used = False
        if not selected and self.fallback_to_top1:
            selected = [(int(sorted_ids[0].item()), float(sorted_probs[0].item()))]
            fallback_used = True

        task_confidences = {task_id: conf for task_id, conf in selected}
        categories = [self.id_to_category.get(task_id, "Unknown") for task_id in task_confidences]
        confidence = max(task_confidences.values()) if task_confidences else 0.0

        if enable_timing:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_time_ms = (time.time() - start_time) * 1000.0
            self._record_stats(inference_time_ms, flops)
            stats = {
                "inference_time_ms": inference_time_ms,
                "flops": flops,
                "min_confidence": self.min_confidence,
                "max_tasks": self.max_tasks,
                "num_selected_tasks": len(task_confidences),
                "top_candidates": [
                    {
                        "task_id": int(task_id.item()),
                        "category": self.id_to_category.get(int(task_id.item()), "Unknown"),
                        "confidence": float(prob.item()),
                    }
                    for prob, task_id in zip(sorted_probs[:2], sorted_ids[:2])
                ],
                "selected_tasks": [
                    {
                        "task_id": task_id,
                        "category": self.id_to_category.get(task_id, "Unknown"),
                        "confidence": conf,
                    }
                    for task_id, conf in selected
                ],
                "fallback_used": fallback_used,
            }

        return task_confidences, confidence, categories, stats

    def predict(
        self, prompt: str, enable_timing: bool = False
    ) -> Tuple[Dict[int, float], float, List[str], Optional[Dict]]:
        return self.predict_multi(prompt, enable_timing=enable_timing)
