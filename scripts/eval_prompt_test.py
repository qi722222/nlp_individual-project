"""Test different inference prompts on Round 3 adapter WITHOUT retraining.
Compares the original prompt vs a 'complete phrase' prompt.
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


def load_cfg():
    return yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())


def generate(model, tokenizer, rows, prompt_template, max_new_tokens=32, bs=8):
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for start in tqdm(range(0, len(rows), bs), desc="generate"):
        batch = rows[start:start + bs]
        prompts = [prompt_template.format(question=r["question"]) for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


def score(rows, gens):
    strict = sum(1 for r, g in zip(rows, gens) if r["correct_answer"].strip().lower() in g.strip().lower())
    return strict, len(rows), strict / len(rows)


def main():
    cfg = load_cfg()
    model_path = REPO_ROOT / cfg["model_path"].lstrip("./")

    # Try Round 3 adapter first, fall back to current output
    adapter_dir = REPO_ROOT / "output_round3" / "adapter"
    if not adapter_dir.exists():
        adapter_dir = REPO_ROOT / "output" / "adapter"
    print(f"[test] using adapter: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    val = json.loads((REPO_ROOT / "data" / "val.json").read_text())

    prompts = {
        "original": "Question: {question}\nAnswer:",
        "complete_phrase": "Question: {question}\nAnswer with a complete phrase:",
        "full_sentence": "Question: {question}\nPlease answer the following question with a full and complete answer:",
    }

    for name, tmpl in prompts.items():
        print(f"\n=== Prompt: {name} ===")
        print(f"    Template: {tmpl[:60]}...")
        gens = generate(model, tokenizer, val, tmpl)
        correct, total, acc = score(val, gens)
        print(f"    Val strict accuracy: {acc:.4f} ({correct}/{total})")
        # Show first 5 examples
        for i in range(5):
            print(f"    [{i}] gold={val[i]['correct_answer']!r}  pred={gens[i].strip()!r}")


if __name__ == "__main__":
    main()
