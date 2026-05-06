import json
import argparse
from pathlib import Path


def convert_mme_jsonl(input_file: str, output_file: str):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("a", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            instruction = item.get("prompt", "").strip()
            if not instruction:
                continue

            count += 1

            new_item = {
                "id": f"ScineceQA_{count:06d}",
                "source": "ScineceQA",
                "instruction": instruction
            }

            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    print(f"Done. Converted {count} samples.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/cwd/codes/SparseVLMs-1.5/playground/data/eval/scienceqa/answers/llava-v1.5-7b-192.jsonl", help="Path to original MME jsonl file")
    parser.add_argument("--output", default="/home/cwd/codes/SparseVLMs-1.5/text/text.jsonl", help="Path to output jsonl file")

    args = parser.parse_args()

    convert_mme_jsonl(args.input, args.output)