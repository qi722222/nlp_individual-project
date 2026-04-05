# AIAA 4051: Introduction to NLP — Individual Project Requirements

**Course:** AIAA 4051 Introduction to NLP, Artificial Intelligence Thrust, HKUST(GZ), Spring 2026
**Deadline:** 23:55, April 15

---

## 1. Task: Fine-tuning Llama 2-7B with LoRA

### 1.1 Task Description
- Use the **Llama 2-7B** model and fine-tune it on the provided dataset using **LoRA**.
- Dataset location: Canvas → `Files/dataset`.
- Each student splits the dataset into **training set** and **validation set** on their own.
- During training, compute and store the model's **accuracy**, defined as:
  > accuracy = (# times the correct answer appears in the generated response) / (total # of questions)

### 1.2 LoRA Requirement
- Must use **PEFT** to implement LoRA fine-tuning for Llama 2-7B.
- Reference docs:
  - PEFT: https://huggingface.co/docs/peft/index
  - LoRA in PEFT: https://huggingface.co/docs/peft/developer_guides/lora

---

## 2. Submission Requirements

### 2.1 PDF Report — must include:
1. **Detailed description of training process**: dataset splitting, model setup, fine-tuning procedure, evaluation steps.
2. **Detailed summary of training hyperparameters**, especially LoRA-related ones: `rank`, `alpha`, `dropout`, `epochs`, plus other important training settings.
3. **Plots** of the **training loss curve** and the **validation loss curve**.
4. **Accuracy** evaluated on both the **training set** and the **validation set**.

### 2.2 Source Code
- Must include PEFT-based LoRA fine-tuning code.

### 2.3 Fine-tuned Model
- Only the **LoRA adapter files** need to be saved/submitted.
- **Do NOT** upload the original Llama 2-7B weights.

### 2.4 Submission Folder Structure
All files go in a single folder named `studentID_name` (e.g. `50011190_Yazheng Liu`):
```
studentID_name/
├── studentID_name.pdf              # the report
├── studentID_name_code/            # source code
└── studentID_name_model/           # saved LoRA adapter files
```

---

## 3. Evaluation
- Final grade is based on:
  - Model's **accuracy on an additional test set** (not provided to students; the instructor runs the submitted fine-tuned model on it).
  - Completeness, clarity, and quality of the report.

---

## 4. Additional Notes
- **Model download**: Llama 2-7B available at ModelScope: https://www.modelscope.cn/models/shakechen/Llama-2-7b
- **Diandong platform**: usage instructions provided via link in the PDF.
- **Code understanding**: part of the project code may appear in the final exam.

---

## 5. Working Environment & Workflow

### 5.1 Constraints
- **Local machine (user's Mac):** only ~20 GB free disk — **NOT** enough for Llama 2-7B (~13 GB) + training artifacts. Do **not** download model weights or run training locally.
- **VM (Diandong or similar):** has GPU + disk. All heavy work (model download, training, evaluation) runs here.
- **License:** Llama 2 weights cannot be redistributed (Meta license) — they must be downloaded on the VM directly from ModelScope, never committed to GitHub.

### 5.2 Division of Labor
| Layer | Location | Contents |
|---|---|---|
| GitHub repo | Synced Mac ↔ VM | Source code, `requirements.txt`, `download_model.sh`, dataset (if small), `CLAUDE.md`, `README.md` |
| VM only (gitignored) | VM local disk | Llama-2-7b weights, LoRA checkpoints during training, wandb/tensorboard logs |
| Submitted to course | Zipped `studentID_name/` | Report PDF, code copy, final LoRA adapter files only |

### 5.3 Iteration Loop
1. Claude (on Mac) writes/edits code locally in this folder.
2. User commits & pushes to GitHub.
3. User `git pull`s on the VM and runs the training/eval script.
4. User pastes back logs, loss values, errors, accuracy numbers (NOT large files) to Claude.
5. Claude reads the pasted output, diagnoses, and iterates on the code.
6. Final LoRA adapter files are downloaded from VM → packaged into submission folder.

### 5.4 What Claude Can Do Without the Model
Claude works purely by reading code, configs, and user-pasted output. Claude does NOT need to:
- execute training locally,
- have the model weights on disk,
- access the VM directly.

User is Claude's eyes/hands on the VM. Paste verbatim output (error tracebacks, loss per step, sample generations) for accurate diagnosis.

### 5.5 Info Claude Needs from User (still pending)
- [ ] VM specs: OS, Python version, GPU model + VRAM, CUDA version
- [ ] Whether VM can reach modelscope.cn / huggingface.co
- [ ] Dataset file name, format (JSON? CSV?), size, and a few sample rows — user to download from Canvas and share
- [ ] GitHub repo URL once created

---

## 6. Claude Operating Rules

### 6.1 Shell commands: always single-line
When giving the user shell commands to run on the VM (or locally), **always write each command as a single line**. Do NOT use backslash (`\`) line continuations, even for long commands. The user's terminal / SSH session breaks backslash-continued commands — the second line gets interpreted as a separate command and fails.

**Wrong:**
```
python script.py \
    --input_dir /path/a \
    --output_dir /path/b
```

**Right:**
```
python script.py --input_dir /path/a --output_dir /path/b
```

This applies to `wget`, `curl`, `python`, `pip`, `git`, and all other commands. If a command is long, it stays long on one line.

**Additionally:** if a command exceeds ~80 characters, the terminal visually wraps it and the user's copy-paste re-inserts newlines mid-command. There is no shell character that prevents this. The fix is to **put long commands into a script file in `scripts/`**, commit + push, and give the user a short command like `bash scripts/foo.sh` or `python scripts/foo.py` to run. Never give the user multi-argument `python -c "..."` one-liners or long URL commands to paste.

---

## Checklist Before Submission
- [ ] Dataset split into train/val
- [ ] LoRA fine-tuning implemented via PEFT
- [ ] Training + validation loss tracked & plotted
- [ ] Accuracy computed on train and val sets
- [ ] LoRA adapter files saved (not full model weights)
- [ ] Report PDF with all 4 required sections
- [ ] Folder structure follows `studentID_name/{pdf, _code/, _model/}` convention
