# ============================================
# ============================================

import argparse
import torch
import os
import json
import time
import sys
import contextlib
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
import logging
import datetime
from PIL import Image
import math
import statistics


class Tee:
    """Write stdout/stderr to both console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()
from llava.visualization import PruningMaskVisualizer
from llava.classifier import PromptTaskClassifier, PromptTaskClassifierV2, CATEGORY_MAPPING, ID_TO_CATEGORY


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def summarize_min_max_avg(values):
    if not values:
        return {"min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": float(statistics.mean(values)),
    }

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Load model and optional prompt classifier, then run VQA evaluation.
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    classifier = None
    if args.classifier_path:
        print(f"\n{'='*80}")
        print(f"Loading prompt classifier from {args.classifier_path}")
        try:
            classifier_cls = PromptTaskClassifierV2 if args.classifier_version == "v2" else PromptTaskClassifier
            classifier_kwargs = {
                "model_path": args.classifier_path,
                "num_classes": 14,
            }
            if args.classifier_version == "v2":
                classifier_kwargs.update({
                    "min_confidence": args.classifier_min_confidence,
                    "max_tasks": args.classifier_max_tasks,
                    "fallback_to_top1": not args.classifier_no_fallback,
                })
            classifier = classifier_cls(**classifier_kwargs)
            print("Classifier loaded successfully!")
            if args.classifier_version == "v2":
                print(
                    "  Will use classifier v2.0 multi-task routing "
                    f"(min_confidence={args.classifier_min_confidence}, "
                    f"max_tasks={args.classifier_max_tasks})"
                )
            else:
                print(f"  Will use task-specific attention heads for better performance")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Failed to load classifier: {e}")
            print(f"  Will use default attention head")
            print(f"{'='*80}\n")
            classifier = None

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    print(f"Model device: {device}")

    if classifier is not None:
        classifier.device = device
        classifier.model.to(device)
        print(f"Classifier moved to device: {device}")
    classification_stats = {
        'total': 0,
        'correct': 0,
        'by_category': {cat: {'total': 0, 'correct': 0} for cat in ID_TO_CATEGORY.values()}
    }

    profile_samples = []
    if args.visualize_pruning:
        visualizer = PruningMaskVisualizer(
            image_size=336,
            patch_size=14,
            num_patches_per_side=24,  # 576 = 24*24
        )
        os.makedirs(args.visualization_output_dir, exist_ok=True)
        print(f"Pruning visualization enabled. Output directory: {args.visualization_output_dir}")


    retained_tokens = args.retained_tokens

    for sample_idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions))):
        idx = line["question_id"]
        idx_clean = str(idx).replace("/", "_").replace("\\", "_")
        if idx_clean.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            idx_clean = idx_clean.rsplit('.', 1)[0]
        cur_prompt = line["text"]
        true_category = line.get("category", "Unknown")

        unique_sample_id = f"sample_{sample_idx:05d}_{idx_clean}"

        predicted_task_id = None
        predicted_category = None
        confidence = 0.0

        if classifier is not None:
            try:
                predicted_task_id, confidence, predicted_category, _ = classifier.predict(cur_prompt, enable_timing=False)

                predicted_categories = predicted_category if isinstance(predicted_category, list) else [predicted_category]
                classification_stats['total'] += 1
                if true_category in predicted_categories:
                    classification_stats['correct'] += 1

                if true_category in classification_stats['by_category']:
                    classification_stats['by_category'][true_category]['total'] += 1
                    if true_category in predicted_categories:
                        classification_stats['by_category'][true_category]['correct'] += 1

            except Exception as e:
                print(f"Classifier error for question {idx}: {e}")

        input_ids = input_ids.to(device=device, non_blocking=True)

        if args.visualize_pruning:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            original_image = Image.open(image_path).convert('RGB')

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True),
                image_sizes=image_sizes,
                retained_tokens=retained_tokens,
                task_id=predicted_task_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            sample_profile = {}
            if hasattr(model, "model") and hasattr(model.model, "last_profile"):
                sample_profile = dict(model.model.last_profile)
            if sample_profile:
                sample_profile = {
                    "prefill_time_ms": float(sample_profile.get("prefill_time_ms", 0.0))
                }
                profile_samples.append(sample_profile)

                pruning_masks = model.model.pruning_masks
                pruning_masks_info = model.model.pruning_masks_info
                layer_2_mask = pruning_masks.get(2)
                if layer_2_mask is not None:
                    layer_2_info = pruning_masks_info.get(2, {})
                    topk_retained = layer_2_info.get('topk_retained')
                    image_shape = layer_2_info.get('image_shape', 576)

                    sample_output_dir = os.path.join(args.visualization_output_dir, unique_sample_id)
                    os.makedirs(sample_output_dir, exist_ok=True)

                    vis_image = visualizer.visualize_layer2(
                        original_image,
                        layer_2_mask,
                        save_path=os.path.join(
                            sample_output_dir,
                            "pruning_vis_layer2.png"
                        ),
                        topk_retained=topk_retained,
                        original_image_shape=image_shape
                    )

                    model.model.collect_pruning_masks = False
                    model.model.pruning_masks = {}
                    model.model.pruning_masks_info = {}

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        result = {
            "question_id": idx,
            "prompt": cur_prompt,
            "clean_prompt": line.get("clean_text", cur_prompt),
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "true_category": true_category,
                "predicted_category": predicted_category,
                "predicted_task_id": predicted_task_id,
                "classification_confidence": confidence,
                "profile": sample_profile if sample_profile else None,
            }
        }

        ans_file.write(json.dumps(result) + "\n")

    ans_file.close()

    prefill_times = [float(x.get("prefill_time_ms", 0.0)) for x in profile_samples]
    prefill_time_stats = summarize_min_max_avg(prefill_times)

    print(f"\n{'='*80}")
    print("INFERENCE STATISTICS")
    print(f"{'='*80}")

    print("\n[Efficiency]")
    print(
        f"  Prefill time (ms) min/max/avg: "
        f"{prefill_time_stats['min']:.2f} / "
        f"{prefill_time_stats['max']:.2f} / "
        f"{prefill_time_stats['avg']:.2f}"
    )

    if classifier is not None and classification_stats['total'] > 0:
        print(f"\n{'='*80}")
        print("CLASSIFICATION STATISTICS")
        print(f"{'='*80}")

        overall_acc = 100 * classification_stats['correct'] / classification_stats['total']
        print(f"Overall Accuracy: {overall_acc:.2f}% ({classification_stats['correct']}/{classification_stats['total']})")
        print(f"\nPer-Category Accuracy:")
        print(f"{'Category':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
        print(f"{'-'*60}")

        for cat_name in sorted(classification_stats['by_category'].keys()):
            stats = classification_stats['by_category'][cat_name]
            if stats['total'] > 0:
                acc = 100 * stats['correct'] / stats['total']
                print(f"{cat_name:<30} {stats['correct']:<10} {stats['total']:<10} {acc:.2f}%")

        print(f"{'='*80}\n")

        stats_file = answers_file.replace('.jsonl', '_classification_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(classification_stats, f, indent=2)
        print(f"Classification statistics saved to: {stats_file}")

    inference_stats_file = answers_file.replace('.jsonl', '_inference_stats.json')
    inference_stats_output = {
        'prefill_time_ms': prefill_time_stats,
    }

    with open(inference_stats_file, 'w') as f:
        json.dump(inference_stats_output, f, indent=2)
    print(f"Inference statistics saved to: {inference_stats_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--retained_tokens", type=int, default=192)

    parser.add_argument("--classifier-path", type=str, default=None,
                        help="Path to the trained classifier model")
    parser.add_argument("--classifier-version", type=str, default="v1", choices=["v1", "v2"],
                        help="Use v1 single-task classifier or v2 threshold-based multi-task router")
    parser.add_argument("--classifier-min-confidence", type=float, default=0.20,
                        help="V2: minimum softmax confidence required for a task route")
    parser.add_argument("--classifier-max-tasks", type=int, default=None,
                        help="V2: optional maximum number of task routes; does not force exactly this many")
    parser.add_argument("--classifier-no-fallback", action="store_true",
                        help="V2: disable fallback to top-1 when no task reaches the confidence threshold")

    parser.add_argument("--visualize-pruning", action="store_true",
                        help="Enable pruning mask visualization")
    parser.add_argument("--visualization-output-dir", type=str, 
                        default="./playground/data/eval/MME/visualizations",
                        help="Directory to save visualization results")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to save test run logs. Default: <answers_file>_run.log")

    args = parser.parse_args()
    answers_file = os.path.expanduser(args.answers_file)
    default_log_file = answers_file.replace('.jsonl', '_run.log')
    log_file = os.path.expanduser(args.log_file) if args.log_file else default_log_file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as log_f:
        tee_out = Tee(sys.stdout, log_f)
        tee_err = Tee(sys.stderr, log_f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"Log file: {log_file}")
            eval_model(args)
