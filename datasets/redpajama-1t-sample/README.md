---
pretty_name: RedPajama Data 1T Sample Backup
language:
- en
license: apache-2.0
task_categories:
- text-generation
tags:
- redpajama
- language-modeling
- backup
- parquet
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train-*.parquet"
---

# RedPajama Data 1T Sample Backup

This dataset is a backup mirror of `togethercomputer/RedPajama-Data-1T-Sample`.

It is provided for easier access when the original dataset is unavailable or difficult to download.

## Usage

Original:

```python
from datasets import load_dataset

ds = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
````

Backup:

```python
from datasets import load_dataset

ds = load_dataset(
    "ll922/RedPajama-Data-1T-Sample-Backup",
    split="train",
)
```

## Data

The data is stored in Parquet format under the `data/` directory.

## Source

Original dataset: `togethercomputer/RedPajama-Data-1T-Sample`