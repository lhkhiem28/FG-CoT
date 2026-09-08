import os
import time
import tqdm
import torch
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings("ignore")

from source.config import parse_args_llm
from source.utils.help_funcs import seed_everything
from source.utils.help_funcs import collate_fn
from source.utils.help_funcs import _save_checkpoint, _reload_model
from source.datasets import *
from source.models import *
from source.utils.evaluation import *

def listize_fn(original_batch):
    batch = {}
    for k in original_batch.keys():
        batch[k] = [original_batch[k]]
    return batch

def main(args):
    seed = args.seed
    seed_everything(seed=seed)

    # Step 1: Build dataset
    test_dataset = load_dataset[args.dataset](path = args.path, prop = args.prop, split = "test")
    if "&" not in args.prop:
        test_dataset = [item for item in test_dataset if item["codes"] != "=0"]
    if args.test_ratio < 1.0:
        # sampled from args.seed only, so the subset is the same across models/checkpoints
        n_samples = max(1, round(args.test_ratio*len(test_dataset)))
        indices = sorted(random.Random(seed).sample(range(len(test_dataset)), n_samples))
        test_dataset = [test_dataset[index] for index in indices]
        print(f'Evaluating on {n_samples} test items ({100*args.test_ratio:.0f}% subset, seed {seed})')

    # Step 2: Build model
    args.llm_path = get_llm_path[args.llm_name]
    model = load_model[args.model_name](args=args)
    if args.checkpoint_path is not None:
        model = _reload_model(model, args.checkpoint_path)

    # Step 3: Evaluating
    model.eval()
    eval_outputs = []
    progress_bar_test = tqdm.tqdm(range(len(test_dataset)))

    for index in range(len(test_dataset)):
        batch = test_dataset[index]
        with torch.no_grad():
            output = model.inference(listize_fn(batch))
            eval_outputs.append(output)

        progress_bar_test.update(1)

    # Step 4: Post-processing & report
    eval_funcs[args.dataset](eval_outputs, args)

if __name__ == "__main__":
    args = parse_args_llm().parse_args()
    main(args)