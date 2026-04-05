"""PEFT LoRA fine-tuning of Llama-2-7b on the science-QA dataset.

Run from the repo root:
    python scripts/train.py
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
IGNORE_INDEX = -100


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "lora_config.yaml").read_text())


def build_example(example: Dict[str, str], tokenizer, max_len: int) -> Dict[str, list]:
    """Tokenize a single (question, answer) example for causal LM with loss masked on the prompt."""
    prompt = PROMPT_TEMPLATE.format(question=example["question"]) + " "
    answer = example["correct_answer"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # Prepend BOS
    input_ids = [tokenizer.bos_token_id] + prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * (1 + len(prompt_ids)) + answer_ids

    # Truncate from the right
    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attention_mask = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_token_id] * pad)
            labels.append(x["labels"] + [IGNORE_INDEX] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, tokenizer, max_len: int):
        self.rows = json.loads(path.read_text())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return build_example(self.rows[idx], self.tokenizer, self.max_len)


def main():
    cfg = load_cfg()
    model_path = REPO_ROOT / cfg["model_path"].lstrip("./")
    output_dir = REPO_ROOT / cfg["output_dir"].lstrip("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[train] loading base model from {model_path}")
    torch_dtype = torch.bfloat16 if cfg["bf16"] else (torch.float16 if cfg["fp16"] else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JSONDataset(REPO_ROOT / cfg["train_path"].lstrip("./"), tokenizer, cfg["max_seq_length"])
    val_ds = JSONDataset(REPO_ROOT / cfg["val_path"].lstrip("./"), tokenizer, cfg["max_seq_length"])
    print(f"[train] train size={len(train_ds)}  val size={len(val_ds)}")

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print("[train] starting training ...")
    trainer.train()

    # Save final adapter + tokenizer + training log
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (output_dir / "trainer_log.json").write_text(
        json.dumps(trainer.state.log_history, ensure_ascii=False, indent=2)
    )
    print(f"[train] adapter saved to {adapter_dir}")
    print(f"[train] log saved to {output_dir / 'trainer_log.json'}")


if __name__ == "__main__":
    main()
