"""Diagnostic: check whether our split-then-concat tokenization matches
what the tokenizer produces for the full string. If they differ, it means
we're training the model to predict a different token sequence than what
appears at inference time — a real bug.
"""
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "model" / "Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

PROMPT_TEMPLATE = "Question: {question} Answer:"

examples = [
    {"question": "What type of molecules are found in the hydration shell of a dissolved ion?", "correct_answer": "water"},
    {"question": "Did darwin believe in punctuated equilibrium or gradualism?", "correct_answer": "gradualism"},
    {"question": "Gene expression and what else are usually considered the same molecular process?", "correct_answer": "protein synthesis"},
]

print(f"BOS = {tokenizer.bos_token_id}  EOS = {tokenizer.eos_token_id}  PAD = {tokenizer.pad_token_id}")
print("=" * 80)

for ex in examples:
    q, a = ex["question"], ex["correct_answer"]

    # ----- OUR METHOD (current build_example) -----
    prompt = PROMPT_TEMPLATE.format(question=q) + " "
    answer = a + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    ours = [tokenizer.bos_token_id] + p_ids + a_ids

    # ----- GROUND TRUTH (tokenize the whole thing at once) -----
    full = PROMPT_TEMPLATE.format(question=q) + " " + a + tokenizer.eos_token
    gt = tokenizer(full, add_special_tokens=False)["input_ids"]
    gt = [tokenizer.bos_token_id] + gt

    # ----- INFERENCE TIME (teacher test) -----
    inf_prompt = PROMPT_TEMPLATE.format(question=q)  # no trailing space!
    inf_ids = tokenizer(inf_prompt, add_special_tokens=False)["input_ids"]
    inf_ids = [tokenizer.bos_token_id] + inf_ids

    print(f"\nQ: {q[:60]}...")
    print(f"A: {a!r}")
    print(f"  split-then-concat ({len(ours)} toks): {ours}")
    print(f"  full-tokenize     ({len(gt)} toks): {gt}")
    print(f"  inference prompt  ({len(inf_ids)} toks): {inf_ids}")
    if ours == gt:
        print("  MATCH: split-then-concat == full-tokenize")
    else:
        print("  *** MISMATCH *** split-then-concat != full-tokenize")
        # Show where they diverge
        for i, (o, g) in enumerate(zip(ours, gt)):
            if o != g:
                print(f"    first diff at idx {i}: ours={o} ({tokenizer.decode([o])!r}) vs gt={g} ({tokenizer.decode([g])!r})")
                break

    # Check: does training sequence prefix (prompt part) match inference prompt?
    if ours[: len(inf_ids)] == inf_ids:
        print("  MATCH: training-prompt-prefix == inference-prompt")
    else:
        print("  *** MISMATCH *** training-prompt-prefix != inference-prompt")
        print(f"    train prefix: {ours[: len(inf_ids) + 2]}")
        print(f"    inf prompt:   {inf_ids}")

    # Decode the prompt portion of each to see characters
    print(f"  train-prompt decoded: {tokenizer.decode(ours[: len(p_ids) + 1])!r}")
    print(f"  inf-prompt   decoded: {tokenizer.decode(inf_ids)!r}")
