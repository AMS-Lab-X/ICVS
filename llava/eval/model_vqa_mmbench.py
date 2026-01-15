import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

# 🔥 导入分类器
from llava.classifier import PromptTaskClassifier, CATEGORY_MAPPING, ID_TO_CATEGORY


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


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

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    retained_tokens = args.retained_tokens
    
    # 🔥 统计信息
    classification_stats = {
        'total': 0,
        'correct': 0,
        'by_category': {cat: {'total': 0, 'correct': 0} for cat in ID_TO_CATEGORY.values()}
    }

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            
            # 🔥 获取真实类别（如果数据中有）
            true_category = row.get('category', 'Unknown') if hasattr(row, 'get') else 'Unknown'
            
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            
            # 🔥 使用分类器预测任务类型
            predicted_task_id = None
            predicted_category = None
            confidence = 0.0
            
            if classifier is not None:
                try:
                    predicted_task_id, confidence, predicted_category, _ = classifier.predict(cur_prompt)
                    
                    # 统计分类准确率（只在第一轮统计，避免重复计数）
                    if round_idx == 0:
                        classification_stats['total'] += 1
                        if predicted_category == true_category:
                            classification_stats['correct'] += 1
                        
                        if true_category in classification_stats['by_category']:
                            classification_stats['by_category'][true_category]['total'] += 1
                            if predicted_category == true_category:
                                classification_stats['by_category'][true_category]['correct'] += 1
                    
                except Exception as e:
                    print(f"Classifier error for question {idx}: {e}")
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    retained_tokens=retained_tokens,
                    task_id=predicted_task_id,  # 🔥 传递任务ID到模型
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            result = {
                "question_id": idx,
                "round_id": round_idx,
                "prompt": cur_prompt,
                "text": outputs,
                "options": options,
                "option_char": cur_option_char,
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
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    
    ans_file.close()
    
    # 🔥 打印统计结果
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
        
        # 保存统计结果
        stats_file = answers_file.replace('.jsonl', '_classification_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(classification_stats, f, indent=2)
        print(f"Classification statistics saved to: {stats_file}")


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
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--retained_tokens", type=int, default=192)
    
    # 🔥 分类器参数
    parser.add_argument("--classifier-path", type=str, default=None,
                        help="Path to the trained classifier model")
    
    args = parser.parse_args()

    eval_model(args)