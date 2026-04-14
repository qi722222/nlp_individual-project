"""VM-ready copy of the instructor's individual_project_test.py.

- Paths default to this repo's layout on the VM.
- Override any path via CLI flags.
- Writes per-sample predictions to a CSV and prints accuracy.

Usage (on VM, from repo root):
    python scripts/run_teacher_test.py
    python scripts/run_teacher_test.py --test_data data/val.json
    python scripts/run_teacher_test.py --adapter output/adapter --test_data data/val.json --out output/teacher_test_val.csv
"""
import argparse
import csv
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=str(REPO_ROOT / "model" / "Llama-2-7b-hf"))
    p.add_argument("--adapter", default=str(REPO_ROOT / "output" / "adapter"))
    p.add_argument("--test_data", default=str(REPO_ROOT / "data" / "val.json"))
    p.add_argument("--out", default=str(REPO_ROOT / "output" / "teacher_test_results.csv"))
    p.add_argument("--max_new_tokens", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[test] base_model = {args.base_model}")
    print(f"[test] adapter    = {args.adapter}")
    print(f"[test] test_data  = {args.test_data}")
    print(f"[test] out        = {args.out}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")

    correct = 0
    results = []

    for example in tqdm(test_data):
        question = example["question"]
        true_answer = example["correct_answer"].strip().lower()

        # NOTE: exact prompt format used by the instructor's test script.
        prompt = f"Question: {question} Answer:"
        encoding = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        pred_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

        is_correct = true_answer in pred_answer
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "true_answer": example["correct_answer"],
            "pred_answer": pred_answer,
            "is_correct": is_correct,
        })

    accuracy = 100 * correct / len(test_data)
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_data)})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "true_answer", "pred_answer", "is_correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"[test] wrote {len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
