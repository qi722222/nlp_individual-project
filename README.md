# NLP Individual Project — LoRA Fine-tuning of Llama-2-7B

**Student:** 50012431 JiaqiLI — AIAA 4051, HKUST(GZ), Spring 2026

## Setup (on the VM)

```bash
git clone https://github.com/qi722222/nlp_individual-project.git
cd nlp_individual-project
pip install -r requirements.txt
bash scripts/download_model.sh            # ~13 GB, goes to ./model/Llama-2-7b
python scripts/prepare_data.py            # 4500 train / 500 val
python scripts/train.py                   # LoRA fine-tune, saves to ./output/adapter
python scripts/evaluate.py                # accuracy on train + val
python scripts/plot_curves.py             # loss curves -> ./output/loss_curves.png
```

## Layout

```
.
├── CLAUDE.md              # project spec (auto-loaded by Claude Code)
├── configs/lora_config.yaml
├── data/dataset.json      # 5000 science QA samples
├── model/                 # (gitignored) Llama-2-7b weights on VM
├── output/                # (gitignored) checkpoints, adapter, logs, plots
└── scripts/
    ├── download_model.sh
    ├── prepare_data.py
    ├── train.py
    ├── evaluate.py
    └── plot_curves.py
```

## Hyperparameters

See `configs/lora_config.yaml`. Starting point:
LoRA `r=8, alpha=16, dropout=0.05`; 3 epochs, lr 2e-4, effective batch size 16, bf16.
