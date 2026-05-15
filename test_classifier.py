import argparse
import json

from llava.classifier import PromptTaskClassifierV2


DEFAULT_PROMPTS = [
    "What color is the car and how many people are standing beside it?",
    "Read the text on the sign and translate it into English.",
    "Where is the red cup, and is there a laptop on the table?",
    "Identify the landmark in the image and describe the scene.",
    "Who is the celebrity in the poster, and what color is the background?",
    "Count the objects and explain whether the arrangement makes sense.",
    "The image shows a python code. Is the output of the code '7'?",
    "How many red cars are there on the road"
]


def parse_args():
    parser = argparse.ArgumentParser(description="The image shows a python code. Is the output of the code '7'?")
    parser.add_argument("--classifier-path", type=str, default="/home/cwd/codes/SparseVLMs-1.5/checkpoints/prompt_classifier_add/best_model.pth", help="Path to the trained classifier .pth file")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Optional tokenizer path; defaults to classifier directory")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="Minimum confidence for selecting a task")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional max selected tasks; does not force exactly k")
    parser.add_argument("--no-fallback", action="store_true", help="Do not fall back to top-1 if no task passes threshold")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to classify. Can be repeated. If omitted, built-in mixed-task prompts are used.",
    )
    parser.add_argument("--jsonl-output", type=str, default=None, help="Optional path to save results as JSONL")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    classifier = PromptTaskClassifierV2(
        model_path=args.classifier_path,
        tokenizer_path=args.tokenizer_path,
        min_confidence=args.min_confidence,
        max_tasks=args.max_tasks,
        fallback_to_top1=not args.no_fallback,
    )

    output_file = open(args.jsonl_output, "w", encoding="utf-8") if args.jsonl_output else None
    try:
        for idx, prompt in enumerate(prompts, start=1):
            task_confidences, confidence, categories, stats = classifier.predict(prompt, enable_timing=True)
            result = {
                "idx": idx,
                "prompt": prompt,
                "selected_task_ids": list(task_confidences.keys()),
                "selected_task_confidences": task_confidences,
                "selected_categories": categories,
                "max_confidence": confidence,
                "stats": stats,
            }

            print(f"\n[{idx}] {prompt}")
            if categories:
                for task_id, category in zip(result["selected_task_ids"], categories):
                    print(f"  - {task_id:02d} {category}: {task_confidences[task_id]:.4f}")
            else:
                print("  - No task selected")

            if output_file is not None:
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if output_file is not None:
            output_file.close()

    stats = classifier.get_stats()
    print(
        "\nSummary: "
        f"{stats['num_predictions']} prompts, "
        f"avg_time={stats['avg_time_ms']:.2f} ms, "
        f"avg_flops={stats['avg_flops_g']:.6f} GFLOPs"
    )


if __name__ == "__main__":
    main()
