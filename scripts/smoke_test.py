"""Quick smoke test: load the HF-converted Llama-2-7b model and tokenizer.
Does NOT train. Just confirms the weights are usable end-to-end.
"""
from pathlib import Path
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())
model_path = REPO_ROOT / cfg["model_path"].lstrip("./")

print(f"[smoke] loading tokenizer from {model_path}")
tok = AutoTokenizer.from_pretrained(model_path)
print(f"[smoke] tokenizer ok, vocab size = {tok.vocab_size}")

print(f"[smoke] loading model (bf16) ...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"[smoke] model loaded, params = {n_params:.3f}B")
print(f"[smoke] device = {next(model.parameters()).device}")

# Try one forward pass + generate
prompt = "Question: What is the capital of France?\nAnswer:"
inputs = tok(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
print(f"[smoke] generated: {tok.decode(out[0], skip_special_tokens=True)!r}")
print("[smoke] OK")
