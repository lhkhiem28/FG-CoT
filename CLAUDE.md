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
bash inference.sh qwen3-8b
bash inference.sh qwen3-8b 1.0            # full test split

# one setting
python inference.py --llm_name qwen3-8b --prop 'LogP' --test_ratio 0.05 --accuracy_only

# a fine-tuned LoRA checkpoint (also enables CSV output, see below)
python inference.py --llm_name qwen3-8b --prop 'LogP&QED' --llm_frozen False \
  --checkpoint_path output/train/LogP&QED/llm_qwen3-8b_....pth
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
Accuracy@0.5 requires every property code satisfied **and** Morgan/Tanimoto similarity ≥ 0.5 to
the input; unparseable predictions count as validity misses and stay in the accuracy denominator.

### Prompting bypasses chat templates

`llm.py` never calls `apply_chat_template`. It hardcodes `BOS`/`EOS_USER`/`EOS` per model family,
dispatched on substrings of the LLM path, and feeds `inputs_embeds` — embeddings looked up
manually, left-padded — rather than `input_ids`. The split exists because the prompt is truncated
to `--max_prompt_length` *between* `BOS` and `EOS_USER`.

Consequence: **anything the chat template would do has to be written into these strings by hand.**
That is how thinking is disabled for the hybrid Qwen3/Qwen3.5 models — their `EOS_USER` ends with
an empty think block, the exact text `apply_chat_template(..., enable_thinking=False)` emits after
the assistant header, so generation starts in non-thinking mode:

```python
# Qwen3 / Qwen3.5 — checked first, since "Qwen3" also matches "Qwen3.5"
EOS_USER = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
# other Qwen (2.x)
EOS_USER = '<|im_end|>\n<|im_start|>assistant\n'
```

Verify markers against the real template rather than assuming — render
`apply_chat_template([{"role": "user", "content": SENTINEL}], tokenize=False,
add_generation_prompt=True, enable_thinking=False)` and split on the sentinel (only
`tokenizer_config.json` is needed, not the weights). The families have already diverged: Qwen3.5's
*thinking* prefix is `assistant\n<think>\n` where Qwen3's is bare `assistant\n`; only the
non-thinking form is currently identical across the two.

`<think>` / `</think>` are added tokens with `special: False`, so the slow tokenizer encodes them
as single ids (good) but `skip_special_tokens=True` does **not** strip them from output. A model
that ignores the prefill and emits a think block anyway leaves that text in `pred`, where it fails
`MolFromSmiles` and counts as an invalidity — there is no post-processing step.

The dispatch ends in `else: raise ValueError`, so a family added to `get_llm_path` without markers
fails immediately with a clear message instead of an `AttributeError` deep in `forward()`.

### Inference details

- Batch size is fixed at 1: the loop wraps a single item with `listize_fn`. `collate_fn` in
  `help_funcs.py` is for the (absent) DataLoader/training path.
- Two decode passes: greedy for `pred`, plus a sampled pass producing `|`-joined `generations`
  used for the diversity metric — only when `--num_return_sequences > 1` and *not*
  `--accuracy_only`.
- `--accuracy_only` reports Validity and Accuracy@0.5 only, skipping Novelty/Diversity/SA and the
  sampled pass. Novelty needs `{--split}.json` (default `train`) as the reference set; SA needs
  RDKit's `Contrib/SA_Score`. Both degrade to `N/A` rather than failing.
- Per-prediction CSV is written **only when `--checkpoint_path` is set** (`.pth` → `.csv`), so
  zero-shot runs print metrics and persist nothing.

### Reproducibility

`seed_everything(args.seed)` at startup, and `--test_ratio` subsamples the test split with a
dedicated `random.Random(seed)` (not the global RNG, so decoding can't shift the subset) over
indices taken *after* the `=0` filter. Same seed + same `--prop` ⇒ same subset across models and
checkpoints; a different `--prop` is a different subset.
