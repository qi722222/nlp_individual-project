"""Check whether tokenizer produces any token ids >= model.vocab_size."""
import json
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model" / "Llama-2-7b-hf"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
cfg = json.loads((MODEL_DIR / "config.json").read_text())
model_vocab = cfg["vocab_size"]

print(f"model vocab_size      : {model_vocab}")
print(f"tokenizer vocab_size  : {tok.vocab_size}")
print(f"len(tokenizer)        : {len(tok)}")
print(f"bos_token_id          : {tok.bos_token_id}")
print(f"eos_token_id          : {tok.eos_token_id}")
print(f"pad_token_id          : {tok.pad_token_id}")
print(f"unk_token_id          : {tok.unk_token_id}")

train = json.loads((REPO_ROOT / "data" / "train.json").read_text())

max_id = 0
worst = None
for i, row in enumerate(train):
    prompt = f"Question: {row['question']}\nAnswer: "
    answer = row["correct_answer"] + tok.eos_token
    ids_prompt = tok(prompt, add_special_tokens=False).input_ids
    ids_answer = tok(answer, add_special_tokens=False).input_ids
    full = [tok.bos_token_id] + ids_prompt + ids_answer
    m = max(full)
    if m > max_id:
        max_id = m
        worst = (i, row, full)

print(f"\nscanned {len(train)} train samples")
print(f"max token id seen     : {max_id}")
print(f"max >= model vocab?   : {max_id >= model_vocab}")
if max_id >= model_vocab:
    i, row, full = worst
    print(f"\noffending sample idx  : {i}")
    print(f"  question            : {row['question']}")
    print(f"  answer              : {row['correct_answer']}")
    print(f"  ids                 : {full}")
    over = [x for x in full if x >= model_vocab]
    print(f"  out-of-range ids    : {over}")
