# ============================================
# 文件2: llava/eval/model_vqa_loader.py (修改版)
# 集成分类器并传递任务ID
# ============================================

import argparse
import torch
import os
import json
import time
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

# 🔥 导入可视化工具
from llava.visualization import PruningMaskVisualizer

# 🔥 导入分类器
from llava.classifier import PromptTaskClassifier, CATEGORY_MAPPING, ID_TO_CATEGORY



def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # 🔥 初始化分类器
    classifier = None
    if args.classifier_path:
        print(f"\n{'='*80}")
        print(f"Loading prompt classifier from {args.classifier_path}")
        try:
            classifier = PromptTaskClassifier(
                model_path=args.classifier_path,
                num_classes=14
            )
            print("✓ Classifier loaded successfully!")
            print(f"  Will use task-specific attention heads for better performance")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"✗ Failed to load classifier: {e}")
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
    
    # 🔥 获取模型所在的设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    print(f"Model device: {device}")
    
    # 🔥 确保分类器使用相同的设备
    if classifier is not None:
        classifier.device = device
        classifier.model.to(device)
        print(f"Classifier moved to device: {device}")
    
    # 🔥 统计信息
    classification_stats = {
        'total': 0,
        'correct': 0,
        'by_category': {cat: {'total': 0, 'correct': 0} for cat in ID_TO_CATEGORY.values()}
    }
    
    # 🔥 推理时间统计
    total_inference_times = []  # 记录每次完整推理的时间（毫秒）
    classifier_stats = None  # 分类器统计信息
    
    # 🔥 初始化可视化器（如果需要可视化）
    visualizer = None
    if args.visualize_pruning:
        visualizer = PruningMaskVisualizer(
            image_size=336,  # 根据实际情况调整
            patch_size=14,
            num_patches_per_side=24,  # 576 = 24*24
        )
        os.makedirs(args.visualization_output_dir, exist_ok=True)
        print(f"Pruning visualization enabled. Output directory: {args.visualization_output_dir}")
    

    
    retained_tokens = args.retained_tokens
    
    # 🔥 为每个样本生成唯一的ID（使用行号，因为同一个图像可能有多个不同的文本prompt）
    for sample_idx, ((input_ids, image_tensor, image_sizes), line) in enumerate(tqdm(zip(data_loader, questions), total=len(questions))):
        idx = line["question_id"]
        # 🔥 清理idx：移除路径分隔符和扩展名，确保文件名安全
        idx_clean = str(idx).replace("/", "_").replace("\\", "_")
        if idx_clean.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            idx_clean = idx_clean.rsplit('.', 1)[0]  # 移除扩展名
        cur_prompt = line["text"]
        true_category = line.get("category", "Unknown")
        
        # 🔥 生成唯一的样本ID：使用行号（sample_idx）确保每个样本都有唯一标识
        # 格式：sample_{行号}_{清理后的question_id}，例如：sample_0_code_reasoning_0020
        unique_sample_id = f"sample_{sample_idx:05d}_{idx_clean}"
        
        
        # 🔥 开始计时整个推理过程（包括分类器和模型生成）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_start_time = time.time()
        
        # 🔥 使用分类器预测任务类型
        predicted_task_id = None
        predicted_category = None
        confidence = 0.0
        
        if classifier is not None:
            try:
                predicted_task_id, confidence, predicted_category, _ = classifier.predict(cur_prompt, enable_timing=True)
                
                # 统计分类准确率
                classification_stats['total'] += 1
                if predicted_category == true_category:
                    classification_stats['correct'] += 1
                
                if true_category in classification_stats['by_category']:
                    classification_stats['by_category'][true_category]['total'] += 1
                    if predicted_category == true_category:
                        classification_stats['by_category'][true_category]['correct'] += 1
                
            except Exception as e:
                print(f"Classifier error for question {idx}: {e}")
        
        input_ids = input_ids.to(device=device, non_blocking=True)
        
        # 🔥 加载原始图像用于可视化
        original_image = None
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
                task_id=predicted_task_id,  # 🔥 传递任务ID到模型
                collect_pruning_masks=args.visualize_pruning,  # 🔥 启用掩码收集
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            
            # 🔥 可视化剪枝掩码（只显示Layer 2）
            if args.visualize_pruning and original_image is not None:
                pruning_masks = model.model.pruning_masks
                pruning_masks_info = model.model.pruning_masks_info
                # 只可视化Layer 2的mask
                layer_2_mask = pruning_masks.get(2)
                if layer_2_mask is not None:
                    layer_2_info = pruning_masks_info.get(2, {})
                    topk_retained = layer_2_info.get('topk_retained')
                    image_shape = layer_2_info.get('image_shape', 576)
                    
                    # 🔥 为当前样本创建子文件夹（与FFT可视化保持一致）
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
                    
                    # 重置掩码收集
                    model.model.collect_pruning_masks = False
                    model.model.pruning_masks = {}
                    model.model.pruning_masks_info = {}
        

        
        # 🔥 结束计时整个推理过程
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_end_time = time.time()
        total_inference_time_ms = (total_end_time - total_start_time) * 1000.0
        total_inference_times.append(total_inference_time_ms)
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        result = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "true_category": true_category,
                "predicted_category": predicted_category,
                "predicted_task_id": predicted_task_id,
                "classification_confidence": confidence,
            }
        }
        
        ans_file.write(json.dumps(result) + "\n")
    
    ans_file.close()
    
    # 🔥 获取分类器统计信息
    if classifier is not None:
        classifier_stats = classifier.get_stats()
    
    # 🔥 计算整体推理统计信息
    overall_inference_stats = {
        'num_samples': len(total_inference_times),
        'total_time_ms': sum(total_inference_times),
        'avg_time_ms': sum(total_inference_times) / len(total_inference_times) if total_inference_times else 0.0,
        'min_time_ms': min(total_inference_times) if total_inference_times else 0.0,
        'max_time_ms': max(total_inference_times) if total_inference_times else 0.0,
    }
    
    # 🔥 打印统计结果
    print(f"\n{'='*80}")
    print("INFERENCE STATISTICS")
    print(f"{'='*80}")
    print(f"\n【Overall Inference (including classifier)】")
    print(f"  Total samples: {overall_inference_stats['num_samples']}")
    print(f"  Total time: {overall_inference_stats['total_time_ms']:.2f} ms")
    print(f"  Average time: {overall_inference_stats['avg_time_ms']:.2f} ms")
    print(f"  Min time: {overall_inference_stats['min_time_ms']:.2f} ms")
    print(f"  Max time: {overall_inference_stats['max_time_ms']:.2f} ms")
    
    if classifier_stats is not None:
        print(f"\n【Classifier Statistics】")
        print(f"  Total predictions: {classifier_stats['num_predictions']}")
        print(f"  Total time: {classifier_stats['total_time_ms']:.2f} ms")
        print(f"  Average time: {classifier_stats['avg_time_ms']:.2f} ms")
        print(f"  Total FLOPs: {classifier_stats['total_flops_g']:.4f} GFLOPs")
        print(f"  Average FLOPs: {classifier_stats['avg_flops_g']:.6f} GFLOPs")
        
        # 计算分类器占整体推理的比例
        if overall_inference_stats['total_time_ms'] > 0:
            classifier_overhead_ratio = (classifier_stats['total_time_ms'] / overall_inference_stats['total_time_ms']) * 100.0
            print(f"  Overhead ratio: {classifier_overhead_ratio:.2f}%")
        
        # 计算模型生成时间（整体时间 - 分类器时间）
        model_generation_time = overall_inference_stats['total_time_ms'] - classifier_stats['total_time_ms']
        avg_model_time = model_generation_time / overall_inference_stats['num_samples'] if overall_inference_stats['num_samples'] > 0 else 0.0
        print(f"\n【Model Generation (excluding classifier)】")
        print(f"  Total time: {model_generation_time:.2f} ms")
        print(f"  Average time: {avg_model_time:.2f} ms")
    
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
    
    # 🔥 保存推理统计信息
    inference_stats_file = answers_file.replace('.jsonl', '_inference_stats.json')
    inference_stats_output = {
        'overall_inference': overall_inference_stats,
    }
    if classifier_stats is not None:
        inference_stats_output['classifier'] = classifier_stats
        if overall_inference_stats['total_time_ms'] > 0:
            inference_stats_output['classifier_overhead_ratio'] = (classifier_stats['total_time_ms'] / overall_inference_stats['total_time_ms']) * 100.0
            inference_stats_output['model_generation_time_ms'] = overall_inference_stats['total_time_ms'] - classifier_stats['total_time_ms']
    
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
    
    # 🔥 分类器参数
    parser.add_argument("--classifier-path", type=str, default=None,
                        help="Path to the trained classifier model")
    
    # 🔥 可视化参数
    parser.add_argument("--visualize-pruning", action="store_true",
                        help="Enable pruning mask visualization")
    parser.add_argument("--visualization-output-dir", type=str, 
                        default="./playground/data/eval/MME/visualizations",
                        help="Directory to save visualization results")
    
    args = parser.parse_args()
    eval_model(args)