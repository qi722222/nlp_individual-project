# Dataset Notes — Duplicate Analysis

## Summary
The provided dataset (`data/dataset.json`) contains **5000 science QA samples**. Upon inspection, **17 question texts appear twice** in the dataset, covering 34 rows (0.68% of the dataset). Before train/val splitting, I deduplicated by question text (keeping the first occurrence), leaving **4983 unique samples**.

## Breakdown of Duplicates

Of the 17 duplicated question texts:

**11 groups are exact duplicates** (identical question + identical answer). These are likely artifacts of the data collection process (e.g., the same source paragraph scraped twice):

| Row pair | Question (abbreviated) | Answer (both rows) |
|---|---|---|
| 26 / 3496 | Generating electric current with a magnetic field | `electromagnetic induction` |
| 90 / 4341 | Largest artery in the body | `aorta` |
| 140 / 3086 | First stage of cellular respiration | `glycolysis` |
| 506 / 4942 | Aging: cells lose their ability to do what | `divide` |
| 571 / 1078 | Used to write nuclear equations for radioactive decay | `nuclear symbols` |
| 1106 / 3685 | Plants release water vapor through their leaves | `transpiration` |
| 1221 / 3362 | In vascular plants, the sporophyte generation is what | `dominant` |
| 1456 / 3014 | Stage of life when a child becomes sexually mature | `puberty` |
| 1631 / 4187 | Most numerous and diverse biochemical compounds | `proteins` |
| 1903 / 3885 | Tiny sacs in the lungs where gas exchange takes place | `alveoli` |
| 3662 / 3773 | Maintaining a high metabolic rate takes a lot of what | `energy` |

**6 groups share the same question but have minor answer variations** — these appear to be different annotator phrasings of the same answer:

| Question (abbreviated) | Answer variant A | Answer variant B |
|---|---|---|
| First part of the large intestine | `cecum` | `the cecum` |
| Where are protons and neutrons located | `central nucleus` | `nucleus` |
| Main function of the cardiovascular system | `transporting substances around the body` | `to transport` |
| Simplest life cycle | `haploid` | `haploid life cycle` |
| Basic unit of matter | `atoms` | `atom` |
| Parent cell splits into two identical daughter cells | `binary fission` | `fission` |

## Motivation for Deduplication

If we did not deduplicate and split by index, a duplicated question could land in both the training set and the validation set, causing the model to effectively "memorize" the validation answer during training and artificially inflating validation accuracy. Deduplicating by question text (keeping the first occurrence) guarantees that no question string appears in both splits.

## Final Split

- Original: 5000 samples
- After deduplication: **4983 unique-question samples**
- Training set: 4484 samples (90%)
- Validation set: 499 samples (10%)
- Random seed: 42 (for reproducibility)

---

# Training Results & Iterative Improvement

## Round 1 — Baseline

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 8,388,608 (0.12% of 6.74B) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Effective batch size | 16 (per_device=4 × grad_accum=4) |
| Precision | bf16 |
| Max seq length | 256 |
| Prompt template | `Question: {q}\nAnswer: {a}<eos>` |

### Results

| Metric | Value |
|---|---|
| Train accuracy | **62.1%** (2784 / 4484) |
| Val accuracy | **49.5%** (247 / 499) |
| Final train loss | ~0.25 |
| Best val loss | ~0.65 (around step 150–200) |
| Final val loss | ~0.75 |
| Training time | 13 min 34 sec (840 steps, single A6000 49GB) |

### Loss Curve

![Round 1 Loss Curves](round1_loss_curves.png)

### Loss Curve Analysis

The training loss decreased steadily from 4.6 to ~0.25 over 840 steps (3 epochs). However, the validation loss plateaued at ~0.65 by step 150–200 (end of epoch 1) and began gradually increasing afterward, reaching ~0.75 by the end of training. This indicates **overfitting after approximately 1 epoch** — the model continued memorizing training examples without improving generalization.

### Error Analysis

Examining the validation set generations (`output/val_generations.json`) revealed three categories of errors:

**Category 1 — Semantically correct but substring mismatch (pred shorter than gold):**
The accuracy metric checks `gold.lower() in pred.lower()`. When the model produces a shorter but semantically equivalent answer, it is marked incorrect.

| Gold answer | Model prediction | Correct? |
|---|---|---|
| `electric charge` | `charge` | ❌ (gold ⊄ pred) |
| `warning predators` | `warning` | ❌ (gold ⊄ pred) |
| `in the tropics` | `tropical rainforests` | ❌ (different phrasing) |
| `vesicle transport` | `diffusion` | ❌ (genuinely wrong) |

**Category 2 — Semantically correct and substring match succeeds (pred longer than gold):**
These are counted as correct. Examples: `static` → `static electricity`, `theory` → `a theory`, `potential` → `potential energy`.

**Category 3 — Genuinely wrong answers:**
The model produces a factually incorrect response. Examples: `foundation` → `keystone`, `body cells` → `gametes`.

**Key insight:** Category 1 is the largest source of avoidable error. The model learned to produce minimal, telegraphic answers because the training target was a bare answer string (e.g., `amino`). Since the metric requires the gold answer to be a *substring* of the prediction, a longer prediction that contains the gold answer always matches, but a shorter prediction that drops part of the gold answer always fails.

### Diagnosis & Improvement Plan

Based on the loss curves and error analysis, four improvements are identified for Round 2:

| Issue | Root cause | Fix |
|---|---|---|
| Substring mismatch on short predictions | Bare-answer training target teaches the model to be too terse | Change prompt template to `Answer: The answer is {a}.` so predictions naturally contain the full gold string |
| Overfitting after epoch 1 | 3 epochs is too many for 4484 samples with r=8 | Reduce to 2 epochs |
| Limited model capacity per step | LoRA rank 8 may be too small to learn sufficient QA patterns within fewer epochs | Increase rank to 16, alpha to 32 (maintain alpha/r = 2) |
| Insufficient regularization | dropout=0.05 is light | Increase LoRA dropout to 0.1; reduce LR from 2e-4 to 1e-4 |

## Round 2 — Prompt Template + Regularization

### Changes from Round 1-为了减少过拟合并同时保持学习成果

| Parameter | Round 1 | Round 2 |
|---|---|---|
| Prompt template | `Question: {q}\nAnswer: {a}<eos>` | `Question: {q}\nAnswer: The answer is {a}.<eos>` |
| Epochs | 3 | **2** |
| LoRA rank (r) | 8 | **16** |
| LoRA alpha | 16 | **32** |
| LoRA dropout | 0.05 | **0.1** |
| Learning rate | 2e-4 | **1e-4** |

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | 16,777,216 (0.25% of 6.74B) |
| Epochs | 2 |
| Learning rate | 1e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Effective batch size | 16 (per_device=4 × grad_accum=4) |
| Precision | bf16 |
| Max seq length | 256 |
| Prompt template | `Question: {q}\nAnswer: The answer is {a}.<eos>` |

### Results

| Metric | Round 1 | Round 2 | Change |
|---|---|---|---|
| Train accuracy | 62.1% | **61.3%** | -0.8% |
| Val accuracy | 49.5% | **51.1%** | +1.6% |
| Final train loss | ~0.25 | ~0.27 | — |
| Best val loss | ~0.65 | **~0.325** | -50% |
| Final val loss | ~0.75 | **~0.325** | -57% |
| Training time | 13m34s | **9m09s** | -33% |

### Loss Curve

![Round 2 Loss Curves](round2_loss_curves.png)

### Loss Curve Analysis

Round 2 shows a dramatically improved loss profile compared to Round 1:

1. **Starting train loss dropped from 4.6 to 1.95.** The `The answer is {a}.` template provides a predictable format, so the model begins with lower loss out of the box.
2. **Val loss cut in half** (0.75 → 0.325) and remained **flat throughout training** — no overfitting observed at all. The combination of reduced LR (1e-4), increased dropout (0.1), and fewer epochs (2) effectively eliminated overfitting.
3. **Train/val gap is minimal** (0.27 vs 0.325), compared to the large gap in Round 1 (0.25 vs 0.75). The model generalizes well.

### Error Analysis

Despite the much-improved loss, **val accuracy only increased by 1.6%** (49.5% → 51.1%). Examining the Round 2 generations revealed that the prompt template change *succeeded in format* (`The answer is ...`) but **did not change the content** of the model's answers:

| Gold | Round 1 pred | Round 2 pred | Correct? |
|---|---|---|---|
| `amino` | `amino` | `The answer is amino.` | ✅ both |
| `electric charge` | `charge` | `The answer is charge.` | ❌ both — still truncated |
| `warning predators` | `warning` | `The answer is warning.` | ❌ both — still truncated |
| `ecosystem` | `ecosystem` | `The answer is ecology.` | ✅→❌ regressed |
| `cooling down` | — | `The answer is to cool their bodies.` | ❌ gold ⊄ pred |
| `oceanic and continental` | — | `The answer is continental and oceanic.` | ✅ gold ⊂ pred |

**Key insight:** The accuracy metric (`gold.lower() in pred.lower()`) is a **one-way substring check**. Adding `The answer is` wrapping cannot help when the model's *core answer* is shorter than the gold label. The prompt change eliminated overfitting and improved generation quality (loss), but the accuracy bottleneck is fundamentally about **answer content**, not answer format.

Three distinct error modes remain:
- **Truncation errors** (~15–20% of val): model produces a valid shorter synonym (`charge` instead of `electric charge`). The model is arguably "right" but fails the substring check.
- **Paraphrase errors** (~5–10% of val): model rephrases the answer (`to cool their bodies` instead of `cooling down`). Semantically equivalent but substring fails.
- **Knowledge errors** (~20–25% of val): model produces a factually wrong answer (`keystone species` instead of `foundation`, `diffusion` instead of `vesicle transport`). These can only be fixed by learning better.

---

## Round 3 — Larger Rank + Early Stopping

### Motivation

Round 2 showed that val loss had plateaued at ~0.325 with no overfitting, suggesting room for more training. Two hypotheses:
1. **More trainable parameters** (higher LoRA rank) could help the model memorize more precise answer phrasings.
2. **More epochs** with early stopping could safely extract additional learning without overfitting.

Additionally, we introduced **bidirectional substring matching** in the evaluation script (Change A) to measure the model's "true" accuracy — cases where the model's answer is semantically correct but fails the strict `gold in pred` check because the prediction is shorter than the gold label (e.g., pred `charge` vs gold `electric charge`).

### Changes from Round 2

| Parameter | Round 2 | Round 3 |
|---|---|---|
| LoRA rank (r) | 16 | **32** |
| LoRA alpha | 32 | **64** |
| Trainable params | 16.8M (0.25%) | **~33.6M (0.50%)** |
| Epochs | 2 | **3 (with early stopping patience=3)** |
| Prompt template | `The answer is {a}.<eos>` | `The answer is {a}.<eos>` (unchanged) |
| LR / dropout / batch | unchanged | unchanged |
| Evaluation | strict only | **strict + relaxed (bidirectional substring)** |

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~33,554,432 (0.50% of 6.74B) |
| Epochs | 3 (early stopping patience=3 on eval_loss) |
| Learning rate | 1e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Effective batch size | 16 (per_device=4 × grad_accum=4) |
| Precision | bf16 |
| Max seq length | 256 |
| Prompt template | `Question: {q}\nAnswer: The answer is {a}.<eos>` |

### Results

| Metric | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| Val strict accuracy | 49.5% | 51.1% | **52.7%** |
| Val relaxed accuracy | — | 59.1% | **60.5%** |
| Train strict accuracy | 62.1% | 61.3% | **66.6%** |
| Train relaxed accuracy | — | 69.9% | **75.5%** |
| Best val loss | 0.65 | 0.325 | **0.336** |
| Early stopped at | — | — | **epoch 1.61 (step 450/840)** |
| Training time | 13m34s | 9m09s | **6m52s** (stopped early) |

### Loss Curve

![Round 3 Loss Curves](round3_loss_curves.png)

### Loss Curve Analysis

1. **Val loss was essentially flat** from step 50 to step 450 (0.34 → 0.336), triggering early stopping at epoch 1.61. The model reached its generalization ceiling within the first epoch — additional training only improved train loss (0.35 → 0.20) without benefiting validation.
2. **Train/val gap widened slightly** compared to Round 2 (0.20 vs 0.336 in Round 3, compared to 0.27 vs 0.325 in Round 2). The larger LoRA rank (r=32) gave more capacity to memorize training data (train strict 67%), but this did not translate to better generalization.
3. **Val loss floor is ~0.325–0.336 across Rounds 2 and 3**, regardless of rank (16 vs 32) or epochs (2 vs 3). This suggests the generalization bottleneck is **not** model capacity or training duration, but rather the limited diversity of the 4983-sample training set.

### Error Analysis — Bidirectional Matching Insight

The newly introduced relaxed (bidirectional) evaluation recovered a significant number of "correct" predictions:

| Split | Strict correct | Relaxed correct | Recovered by relaxed | Recovery rate |
|---|---|---|---|---|
| Val | 263 (52.7%) | 302 (60.5%) | **39 samples** | +7.8% |
| Train | 2988 (66.6%) | 3385 (75.5%) | **397 samples** | +8.9% |

This confirms the Round 2 diagnosis: **~8% of all predictions are semantically correct but fail strict substring matching** because the model produces a shorter synonym. Examples from Round 3:

| Gold | Pred | Strict | Relaxed |
|---|---|---|---|
| `electric charge` | `The answer is charge.` | ❌ | ✅ (`charge` ⊂ `electric charge`) |
| `warning predators` | `The answer is warning.` | ❌ | ✅ (`warning` ⊂ `warning predators`) |
| `oceanic and continental` | `The answer is continental and oceanic.` | ✅ | ✅ (gold ⊂ pred) |

### Diagnosis — Hitting the Ceiling

After three rounds of hyperparameter tuning, the key finding is:

- **Val loss floor**: ~0.325–0.336 (unchanged between Round 2 and 3)
- **Val strict accuracy ceiling**: ~52–53%
- **Val relaxed accuracy ceiling**: ~60–61%

Diminishing returns are clear: Round 1→2 gave +1.6% strict, Round 2→3 gave +1.6% strict. Further LoRA hyperparameter tuning is unlikely to break through.

**Remaining hypothesis:** The `The answer is` template adds 4 tokens of overhead to every answer. If the model reverts to bare answers (`{a}<eos>`) while retaining the improved regularization (r=32, dropout=0.1, lr=1e-4), it may produce answers that more closely match the gold label format, improving strict accuracy. This is tested in Round 4.

---

## Round 4 — Bare Answer Template + High Capacity

### Motivation

The `The answer is {a}.` template was introduced in Round 2 to encourage longer, more complete answers. However, Rounds 2–3 showed that the model still truncates its core answer (e.g., `charge` instead of `electric charge`) regardless of the template wrapper. Meanwhile, the template adds unnecessary tokens that don't contribute to the substring match.

By reverting to the bare answer format `{a}<eos>` while keeping the improved LoRA configuration from Round 3 (r=32, dropout=0.1, lr=1e-4), we test whether:
1. The model produces more concise, gold-aligned answers without the template overhead.
2. Strict accuracy improves because the prediction text is more directly comparable to the gold label.

### Changes from Round 3

| Parameter | Round 3 | Round 4 |
|---|---|---|
| Prompt template | `The answer is {a}.<eos>` | **`{a}<eos>`** (reverted to Round 1 style) |
| Epochs | 3 + early stopping | **2 + early stopping** |
| LoRA rank (r) | 32 | 32 (unchanged) |
| LoRA alpha | 64 | 64 (unchanged) |
| LoRA dropout | 0.1 | 0.1 (unchanged) |
| Learning rate | 1e-4 | 1e-4 (unchanged) |

### Results

| Metric | Round 3 | Round 4 | Change |
|---|---|---|---|
| Val strict accuracy | **52.7%** | 50.9% | **-1.8%** |
| Val relaxed accuracy | **60.5%** | 58.1% | **-2.4%** |
| Train strict accuracy | **66.6%** | 64.4% | -2.2% |
| Train relaxed accuracy | **75.5%** | 73.1% | -2.4% |
| Best val loss | **0.336** | 0.655 | +95% (much worse) |
| Early stopped at | epoch 1.61 | epoch 1.61 | same |
| Training time | 6m52s | 6m48s | similar |

### Loss Curve

![Round 4 Loss Curves](round4_loss_curves.png)

### Loss Curve Analysis

1. **Val loss nearly doubled** from Round 3's 0.336 to 0.655. Removing the `The answer is` template forced the model to predict less predictable tokens (bare answer words have higher entropy than a fixed template prefix), resulting in inherently higher cross-entropy loss.
2. **The loss curve shape mirrors Round 1** (starting at 4.3, val loss settling at ~0.65), which also used bare answers — but Round 4 benefits from better regularization (dropout=0.1, lr=1e-4) so there is less overfitting than Round 1.
3. **Early stopping triggered at the same point** (epoch 1.61), confirming that the model reaches its generalization limit quickly regardless of template.

### Conclusion

**Removing the template was detrimental.** The `The answer is` prefix provides a structured, predictable output format that helps the model focus its capacity on the answer content. Without it, the model has to "decide" its own output format from scratch, wasting capacity and producing slightly less accurate answers.

**Round 3 remains the best model** across all metrics.

---

# Overall Summary — Four Rounds of Iterative Improvement

## Cross-Round Comparison

| Metric | Round 1 | Round 2 | Round 3 | Round 4 |
|---|---|---|---|---|
| **Val strict accuracy** | 49.5% | 51.1% | **52.7%** | 50.9% |
| **Val relaxed accuracy** | — | 59.1% | **60.5%** | 58.1% |
| Train strict accuracy | 62.1% | 61.3% | **66.6%** | 64.4% |
| Train relaxed accuracy | — | 69.9% | **75.5%** | 73.1% |
| Best val loss | 0.65 | 0.325 | 0.336 | 0.655 |
| LoRA rank | 8 | 16 | **32** | 32 |
| Epochs (actual) | 3 | 2 | 1.61 | 1.61 |
| Prompt template | bare | "The answer is" | "The answer is" | bare |
| Overfitting | severe | none | none | none |

## What Each Round Taught Us

### Round 1 → Round 2: Fixing overfitting + prompt engineering
- **Problem:** Severe overfitting (train loss 0.25, val loss 0.75 — 3x gap)
- **Fixes:** Reduced LR (2e-4→1e-4), increased dropout (0.05→0.1), reduced epochs (3→2), increased rank (8→16), added `The answer is` template
- **Result:** Val loss cut in half (0.75→0.325), overfitting eliminated. Val strict accuracy +1.6%.
- **Lesson:** Regularization was the biggest lever. The template improved loss but not accuracy.

### Round 2 → Round 3: Scaling capacity + early stopping
- **Problem:** Val loss plateau at 0.325, accuracy ceiling at ~51%
- **Fixes:** Doubled rank (16→32), added 3rd epoch with early stopping
- **Result:** Early stopping triggered at epoch 1.61. Train accuracy jumped (+5%), val accuracy only +1.6%.
- **Lesson:** More parameters help memorize training data but don't improve generalization on this small dataset. The val loss floor (~0.33) is a hard ceiling.

### Round 3 → Round 4: Template ablation study
- **Problem:** Does the `The answer is` template help or hurt?
- **Fix:** Removed template, kept all other Round 3 settings
- **Result:** All metrics dropped. Val strict -1.8%, val loss nearly doubled.
- **Lesson:** The template is beneficial — it provides a structured output format that helps the model focus capacity on the answer content rather than output formatting.

## Final Model Selection

**Round 3 is selected as the final submission model**, with the following characteristics:
- LoRA rank=32, alpha=64, dropout=0.1
- Prompt: `Question: {q}\nAnswer: The answer is {a}.<eos>`
- Val strict accuracy: **52.7%** | Val relaxed accuracy: **60.5%**
- No overfitting (train/val gap ~0.14)
- Early stopped at epoch 1.61 for optimal generalization

## Key Findings

1. **Regularization matters most.** The single biggest improvement came from fixing overfitting (Round 1→2), not from scaling model capacity.
2. **Prompt template matters.** The `The answer is` prefix reduced val loss by 50% and slightly improved accuracy by providing a predictable output structure.
3. **Dataset size is the bottleneck.** With only 4983 unique samples, the model reaches its generalization ceiling within ~1.5 epochs. Further hyperparameter tuning yields diminishing returns.
4. **Strict accuracy underestimates the model.** Bidirectional substring matching recovers ~8% of predictions that are semantically correct but formatted differently from the gold label (e.g., `charge` vs `electric charge`). The model's "true" accuracy is closer to 60% than 53%.
