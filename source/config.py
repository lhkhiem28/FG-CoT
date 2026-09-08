import argparse

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
    parser.add_argument("--path", type=str, default='../FG-CoT-datasets/ChEMBL28')
    parser.add_argument("--prop", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_prompt_length", type=int, default=1024)

    # Inference
    parser.add_argument("--test_ratio", type=float, default=1.0, help="Fraction of the test split to evaluate on, sampled with --seed for reproducibility")
    parser.add_argument("--max_completion_length", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_return_sequences", type=int, default=16)
    parser.add_argument("--accuracy_only", action="store_true", help="Skip Novelty/Diversity/SA and the sampled decoding pass; report Validity and Accuracy@0.7 only")

    # Checkpoint
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--checkpoint_path", type=str, default=None)

    return parser