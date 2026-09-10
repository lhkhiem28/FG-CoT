import argparse

def ratio_or_count(value):
    """--test_ratio as a fraction (float, e.g. 0.05) or an exact item count (int, e.g. 200).

    The distinction is made on the literal spelling, not the value: '1' means one item,
    '1.0' means the whole test split.
    """
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a number')
    if any(c in value for c in ".eE"):
        if not 0.0 < number <= 1.0:
            raise argparse.ArgumentTypeError(f'fraction {value} must be in (0.0, 1.0]')
        return number
    count = int(number)
    if count < 1:
        raise argparse.ArgumentTypeError(f'item count {value} must be >= 1')
    return count

def parse_args_llm():
    parser = argparse.ArgumentParser(description="FG-CoT")
    parser.add_argument("--project", type=str, default="FG-CoT")
    parser.add_argument("--seed", type=int, default=0)

    # Model related
    parser.add_argument("--model_name", type=str, default='llm')
    parser.add_argument("--llm_name", type=str)
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--n_gpus", type=int, default=2)

    # Model Training
    parser.add_argument("--dataset", type=str, default='generation')
    parser.add_argument("--path", type=str, default='../FG-CoT-datasets/ChEMBL28-0.7')
    parser.add_argument("--prop", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--icl", action="store_true")

    # Inference
    parser.add_argument("--test_ratio", type=ratio_or_count, default=1.0, help="How much of the test split to evaluate on, sampled with --seed for reproducibility: a fraction if written with a decimal point (0.05, 1.0), an exact number of items if written as an integer (200)")
    parser.add_argument("--max_completion_length", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=16)
    parser.add_argument("--accuracy_only", action="store_true", help="Skip Novelty/Diversity/SA and the sampled decoding pass; report Validity and Accuracy only")

    # Checkpoint
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser