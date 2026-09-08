# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FG-CoT evaluates LLMs on **property-guided molecule editing**: given a SMILES string and a
natural-language instruction ("decrease its LogP value by an amount in (0, 0.5]"), the model
emits a modified SMILES, scored with RDKit. Datasets live *outside* the repo, at
`--path` (default `../FG-CoT-datasets/ChEMBL28`, from
https://huggingface.co/datasets/lhkhiem28/FG-CoT-datasets).

There is no `train.py` in the repo. `BaselineLLM.forward()` and `_save_checkpoint()` exist for a
training loop that is not checked in; only the inference/eval path runs here.

## Commands

```bash
# all five property settings (bash, not PowerShell); 2nd arg is the test subset ratio, default 0.05
bash inference.sh qwen2.5-3b
bash inference.sh qwen2.5-3b 1.0          # full test split

# one setting
python inference.py --llm_name qwen2.5-3b --prop 'LogP' --test_ratio 0.05 --accuracy_only

# a fine-tuned LoRA checkpoint (also enables CSV output, see below)
python inference.py --llm_name qwen2.5-3b --prop 'LogP&QED' --llm_frozen False \
  --checkpoint_path output/train/LogP&QED/llm_qwen2.5-3b_....pth
```

No test suite, linter, or requirements file. Runtime deps: `torch`, `transformers`, `peft`,
`rdkit`, `pandas`, `numpy`, `tqdm`. `--n_gpus` assumes 80 GiB cards (`max_memory` in `llm.py`).
For a quick smoke run use a tiny `--test_ratio` rather than editing the loop.

## Architecture

Everything is wired through four registry dicts; adding a dataset/model/LLM means implementing it
*and* registering it:

| Registry | File | Selected by |
| --- | --- | --- |
| `load_dataset` | `source/datasets/__init__.py` | `--dataset` |
| `load_model` | `source/models/__init__.py` | `--model_name` |
| `get_llm_path` | `source/models/__init__.py` | `--llm_name` (maps to an HF repo id or local path) |
| `eval_funcs` | `source/utils/evaluation.py` | `--dataset` |

`inference.py` is the whole pipeline: build dataset → build model → loop → `eval_funcs[...]`.

### `--prop` drives dataset, prompt, and scoring together

Each JSON record carries per-property `Code_<prop>` / `Text_<prop>` / `Delta_<prop>` fields
(`LogP`, `TPSA`, `QED`, `DRD2`, `GSK3B`, `JNK3`). `--prop` picks which; `&` requests several at
once (`'LogP&TPSA'`), and `GenerationDataset` joins codes with `&` and texts with `" and "`.

Codes are interval strings — `+(0, 0.5]`, `-(1.0, inf)`, `=0` — parsed by `prop_check()`, so
`--prop` must use exactly the same `&` spelling on both sides. **Single-property runs drop `=0`
("keep unchanged") records; multi-property runs keep them** (`inference.py:30`), so item counts
differ by setting.

### Scoring is relative to the *input* molecule

`BaselineLLM.inference()` emits `samples["smiles"]` (the original molecule) under the key
`"label"`, not the dataset's `modifiedSMILES`. This is deliberate: the property delta and the
Tanimoto similarity are both measured against the molecule that was handed to the model.
Accuracy@0.7 requires every property code satisfied **and** Morgan/Tanimoto similarity ≥ 0.7 to
the input; unparseable predictions count as validity misses and stay in the accuracy denominator.

### Prompting bypasses chat templates

`llm.py` hardcodes `BOS`/`EOS_USER`/`EOS` per model family (matched on `"Qwen2.5"` / `"Llama-3"`
in the LLM path) and feeds `inputs_embeds` — embeddings looked up manually, left-padded — rather
than `input_ids`. A new model family whose path matches neither string leaves `self.BOS` unset and
fails at first use, so add its markers when adding to `get_llm_path`.

### Inference details

- Batch size is fixed at 1: the loop wraps a single item with `listize_fn`. `collate_fn` in
  `help_funcs.py` is for the (absent) DataLoader/training path.
- Two decode passes: greedy for `pred`, plus a sampled pass producing `|`-joined `generations`
  used for the diversity metric — only when `--num_return_sequences > 1` and *not*
  `--accuracy_only`.
- `--accuracy_only` reports Validity and Accuracy@0.7 only, skipping Novelty/Diversity/SA and the
  sampled pass. Novelty needs `{--split}.json` (default `train`) as the reference set; SA needs
  RDKit's `Contrib/SA_Score`. Both degrade to `N/A` rather than failing.
- Per-prediction CSV is written **only when `--checkpoint_path` is set** (`.pth` → `.csv`), so
  zero-shot runs print metrics and persist nothing.

### Reproducibility

`seed_everything(args.seed)` at startup, and `--test_ratio` subsamples the test split with a
dedicated `random.Random(seed)` (not the global RNG, so decoding can't shift the subset) over
indices taken *after* the `=0` filter. Same seed + same `--prop` ⇒ same subset across models and
checkpoints; a different `--prop` is a different subset.
