# Dataset Notes — Duplicate Analysis

## Summary
The provided dataset (`data/dataset.json`) contains **5000 science QA samples**. Upon inspection, **17 question texts appear twice** in the dataset, covering 34 rows (0.68% of the dataset). Before train/val splitting, I deduplicated by question text (keeping the first occurrence), leaving **4983 unique samples**.

## Breakdown of Duplicates

Of the 17 duplicated question texts:

**11 groups are exact duplicates** (identical question + identical answer). These are likely artifacts of the data collection process (e.g., the same source paragraph scraped twice).

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
