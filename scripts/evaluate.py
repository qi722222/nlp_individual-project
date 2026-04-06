"""Evaluate the fine-tuned LoRA model's accuracy on train and val sets.

Accuracy definition (per course spec):
    accuracy = (# times correct_answer appears in generated response) / (# questions)

Run from the repo root:
    python scripts/evaluate.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())


def generate_answers(model, tokenizer, rows, max_new_tokens, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for start in tqdm(range(0, len(rows), batch_size), desc="generate"):
        batch = rows[start : start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(question=r["question"]) + " " for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


def compute_accuracy(rows, generations):
    correct = 0
    for row, gen in zip(rows, generations):
        if row["correct_answer"].strip().lower() in gen.strip().lower():
            correct += 1
    return correct / len(rows), correct


def main():
    cfg = load_cfg()
    model_path = REPO_ROOT / cfg["model_path"].lstrip("./")
    output_dir = REPO_ROOT / cfg["output_dir"].lstrip("./")
    adapter_dir = output_dir / "adapter"

    print(f"[eval] loading tokenizer from {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # needed for batched generation

    print(f"[eval] loading base model from {model_path}")
    torch_dtype = torch.bfloat16 if cfg["bf16"] else (torch.float16 if cfg["fp16"] else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map={"": 0}
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    print(f"[eval] loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    train_rows = json.loads((REPO_ROOT / cfg["train_path"].lstrip("./")).read_text())
    val_rows = json.loads((REPO_ROOT / cfg["val_path"].lstrip("./")).read_text())

    max_new_tokens = cfg["max_new_tokens"]
    bs = cfg["per_device_eval_batch_size"]

    print(f"[eval] generating on val ({len(val_rows)}) ...")
    val_gens = generate_answers(model, tokenizer, val_rows, max_new_tokens, bs)
    val_acc, val_correct = compute_accuracy(val_rows, val_gens)

    print(f"[eval] generating on train ({len(train_rows)}) ...")
    train_gens = generate_answers(model, tokenizer, train_rows, max_new_tokens, bs)
    train_acc, train_correct = compute_accuracy(train_rows, train_gens)

    result = {
        "train_accuracy": train_acc,
        "train_correct": train_correct,
        "train_total": len(train_rows),
        "val_accuracy": val_acc,
        "val_correct": val_correct,
        "val_total": len(val_rows),
    }
    print("=" * 50)
    print(json.dumps(result, indent=2))

    (output_dir / "accuracy.json").write_text(json.dumps(result, indent=2))
    # Also save generations for inspection
    (output_dir / "val_generations.json").write_text(
        json.dumps(
            [
                {"question": r["question"], "gold": r["correct_answer"], "pred": g}
                for r, g in zip(val_rows, val_gens)
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[eval] saved accuracy -> {output_dir / 'accuracy.json'}")
    print(f"[eval] saved val generations -> {output_dir / 'val_generations.json'}")


if __name__ == "__main__":
    main()
