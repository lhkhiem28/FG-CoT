# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FG-CoT evaluates LLMs on **property-guided molecule editing**: given a SMILES string and a
natural-language instruction ("decrease its LogP value by an amount in (0, 0.5]"), the model
emits a modified SMILES, scored with RDKit. Datasets live *outside* the repo, at
`--path` (default `../FG-CoT-datasets/ChEMBL28-0.7`, from
https://huggingface.co/datasets/lhkhiem28/FG-CoT-datasets).

There is no `train.py` in the repo. `BaselineLLM.forward()` and `_save_checkpoint()` exist for a
training loop that is not checked in; only the inference/eval path runs here.

## Commands

```bash
# all five property settings (bash, not PowerShell); 2nd arg is --test_ratio, default 500 items
bash inference.sh qwen3-14b
bash inference.sh qwen3-14b 1.0            # full test split
bash inference.sh qwen3-14b 0.05           # 5% of each split

# one setting (--test_ratio defaults to 1.0 in config.py; only inference.sh defaults it to 500)
python inference.py --llm_name qwen3-14b --prop 'LogP' --test_ratio 0.05 --accuracy_only
python inference.py --llm_name qwen3-14b --prop 'LogP' --test_ratio 200  --accuracy_only  # exactly 200 items

# a fine-tuned LoRA checkpoint
python inference.py --llm_name qwen3-14b --prop 'LogP&QED' --llm_frozen False \
  --checkpoint_path output/train/LogP&QED/llm_qwen3-14b_....pth
```

`--llm_name` must be a key of `get_llm_path` (`source/models/__init__.py`): currently
`qwen2.5-14b`, `qwen2.5-32b`, `qwen3-14b`, `qwen3-32b`, `llama-3.1-8b`, `llama-3.1-70b`.

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

Each JSON record carries per-property `Code_<prop>` / `Text_<prop>` / `Delta_<prop>` fields.
In `ChEMBL28-0.7` the properties present are exactly **`LogP`, `TPSA`, `QED`** — no activity
properties (`DRD2`/`GSK3B`/`JNK3`) — and all three are scoreable. `--prop` picks which; `&`
requests several at once (`'LogP&TPSA'`), and `GenerationDataset` joins codes with `&` and texts
with `" and "`.

Adding a property means two things, which fail at different times:

- **The dataset must carry the fields.** `--prop DRD2` builds the dataset object fine (the JSON is
  read without touching per-property keys) and then dies with `KeyError: 'Code_DRD2'` in
  `GenerationDataset.__getitem__`. For a single-property run that is immediate — the `=0` filter at
  `inference.py:31` iterates the dataset — but for a `&` run nothing touches `__getitem__` until
  the eval loop, i.e. *after* the LLM has been loaded onto the GPUs.
- **`prop2prop` in `evaluation.py` must map it to an RDKit descriptor** (`MolLogP`, `TPSA`, `qed`).
  A property that exists in the JSON but not here raises `KeyError` inside `prop_check()`, and
  because that call sits under a bare `except` in `get_scores_generation()` the run does *not* fail
  loudly — it reports every prediction as a validity miss.

Codes are interval strings — `+(0, 0.5]`, `-(1.0, inf)`, `=0` — parsed by `prop_check()`, so
`--prop` must use exactly the same `&` spelling on both sides. The interval edges are per-property
(LogP `0.5/1.0`, TPSA `10/20`, QED `0.1/0.2`), which only matters if you write codes by hand.
**Single-property runs drop `=0` ("keep unchanged") records; multi-property runs keep them**
(`inference.py:31`), so item counts differ by setting: of the 14,746 test records, LogP keeps
13,641, QED 13,599, TPSA only 8,279, while `LogP&TPSA` and `LogP&QED` keep all 14,746.

### Scoring is relative to the *input* molecule

`BaselineLLM.inference()` emits `samples["smiles"]` (the original molecule) under the key
`"label"`, not the dataset's `modifiedSMILES`. This is deliberate: the property delta and the
Tanimoto similarity are both measured against the molecule that was handed to the model.
**Accuracy is property-only right now.** It requires every property code satisfied and nothing
else; the Morgan/Tanimoto ≥ 0.5 gate is still in the file but commented out
(`evaluation.py:130`), which is why the printed label is plain `Accuracy` and not `Accuracy@0.5`.
Un-commenting that one line restores the stricter metric — the similarity it needs is already
computed on the line above. Unparseable predictions count as validity misses and stay in the
accuracy denominator. The similarity number itself is now report-only: computed for every
parseable prediction, averaged, and printed as **Similarity** (unless `--accuracy_only`, which
still computes it and just doesn't print it). It is not stored per-row in the CSV.

Three quirks of `get_scores_generation()` to know before trusting a number:

- **The validity denominator inflates on invalid predictions.** `Chem.MolFromSmiles` returns
  `None` rather than raising, so `validities.append(1)` runs first and the `except` branch appends
  a `0` for the *same* prediction once `prop_check()` trips over the `None` mol (RDKit raises
  `ArgumentError`, not a Python exception you'd expect). An invalid prediction therefore
  contributes two entries to `validities`, `len(validities) > len(df)`, and both Validity and
  Accuracy read low. The mirror-image case: an **empty** prediction parses to a zero-atom mol
  rather than `None`, so every descriptor returns `0.0` and the row counts as *valid* — and
  satisfies a `=0` code. `evaluation.py` already has the right guard — `to_mol()`, which rejects
  both `None` and zero-atom mols — but only `get_diversity()`/`get_sa()` use it; the validity loop
  still calls `Chem.MolFromSmiles` bare. `len(validities)` is also the printed
  "/N valid predictions" denominator for Similarity, SA and Diversity, so those coverage fractions
  understate coverage by the same amount.
- **"Similarity" is similarity to the input, not pairwise.** The print label says *mean pairwise
  Tanimoto similarity*, but the value is the mean of the per-prediction input-vs-prediction
  Morgan/Tanimoto numbers, averaged over parseable predictions only — it says nothing about how
  the predictions relate to each other. Diversity is the pairwise metric.
- **Novelty is measured against the training set, not the predictions.**
  `100*(1 - |preds ∩ train| / |train|)` is dominated by `|train|` and sits near 100% regardless of
  model quality. The per-row `novel` column in the CSV *is* per-prediction and is the useful one.
  The reference set is built by `get_train_molecules()` from **both** the `SMILES` and
  `modifiedSMILES` columns of `{--split}.json`, canonicalized, with the same `=0` filter the test
  split gets — so `novel` means "this molecule appears nowhere in that split, as either input or
  target".

### Prompting bypasses chat templates

`llm.py` never calls `apply_chat_template`. It hardcodes `BOS`/`EOS_USER`/`EOS` per model family,
dispatched on substrings of the LLM path, and feeds `inputs_embeds` — embeddings looked up
manually, left-padded — rather than `input_ids`. The three-marker split exists so the prompt can
be truncated to `--max_prompt_length` *between* `BOS` and `EOS_USER`; note that only `forward()`
(the training path) applies that truncation — `inference()` embeds the full prompt, so
`--max_prompt_length` has no effect on an eval run.

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

The Qwen3.5 half of that branch is written but unexercised — no Qwen3.5 entry is registered in
`get_llm_path` yet — so re-verify the markers when one is added.

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
  `--accuracy_only`. `--num_return_sequences` defaults to 16, so on a run without
  `--accuracy_only` that second pass is on by default and costs 16× the decoding.
- `--accuracy_only` reports Validity and Accuracy only, skipping Similarity/Novelty/
  Diversity/SA and the sampled pass. Novelty needs `{--split}.json` (default `train`) as the
  reference set; SA needs RDKit's `Contrib/SA_Score`. Both degrade to `N/A` rather than failing.
- Per-prediction CSV is written on **every** run — twice, in fact: once right after the DataFrame
  is built and again after the optional `novel`/`diversity`/`sa` columns are added. The path is
  `{--output_dir}/{--split}/{--prop}/{--model_name}_{--llm_name}_llm_frozen{--llm_frozen}_{--split}.csv`,
  which encodes neither `--test_ratio` nor `--checkpoint_path`, so two runs differing only in
  those silently overwrite each other. Vary `--output_dir` for runs you want to keep.
- `--split` does double duty: it names the novelty reference JSON *and* the output subdirectory.
  It stays `train` during a test-split eval, which is why results land under `output/train/...`.

### Reproducibility

`seed_everything(args.seed)` at startup, and `--test_ratio` subsamples the test split with a
dedicated `random.Random(seed)` (not the global RNG, so decoding can't shift the subset) over
indices taken *after* the `=0` filter. Same seed + same `--prop` ⇒ same subset across models and
checkpoints; a different `--prop` is a different subset.

`--test_ratio` is parsed by `ratio_or_count()` in `config.py`, which switches on the **literal
spelling** rather than the value: a decimal point or exponent makes it a fraction of the split
(`0.05`, `1e-2`, `1.0` = everything), a bare integer makes it an exact item count (`200`, and
`1` = a single item — *not* the whole split, which is `1.0`). Out-of-range values are rejected at
parse time (fractions must be in `(0.0, 1.0]`, counts `>= 1`); a count larger than the split is
clamped to the split size. Because `inference.sh` interpolates `$2` verbatim, both forms work
there too; it now defaults to the count form (`500`) so every property setting is scored on the
same number of items — with a fraction the five settings get different item counts, since the
`=0` filter shrinks the three single-property splits but not the two `&` ones.
