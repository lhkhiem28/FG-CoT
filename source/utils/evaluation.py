import os
import re
import json
import tqdm
import numpy as np
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import DataStructs

prop2prop = {
    "LogP" : "MolLogP",
    "TPSA" : "TPSA",
    "QED"  : "qed",
}
prop2func = {}
for prop, func in [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in list(set(prop2prop.values()))]:
    prop2func[prop] = func
prop2func = {k:prop2func[prop2prop[k]] for k in prop2prop.keys()}

def prop_check(mol_pred, mol_label, code, prop):
    prop_pred, prop_label = prop2func[prop](mol_pred), prop2func[prop](mol_label)
    gap = prop_pred - prop_label
    if code == "=0":
        return abs(gap) <= 1e-6
    else:
        sign, (intv_a, intv_b) = code[0], code[2:-1].split(", ")
        intv_a, intv_b = float(intv_a), float(intv_b) if intv_b != "inf" else np.inf
        if sign == "+":
            return gap > 0 and (intv_a < abs(gap) and abs(gap) <= intv_b)
        if sign == "-":
            return gap < 0 and (intv_a < abs(gap) and abs(gap) <= intv_b)

def hits_check(mol_pred, mol_label, code, prop):
    hits = []
    for code, prop in zip(code.split("&"), prop.split("&")):
        hits.append(prop_check(mol_pred, mol_label, code, prop))
    return hits

MOLECULE_TAG = re.compile(r'<molecule>(.*?)</molecule>', re.DOTALL)

def extract_molecule(text):
    """Pull the SMILES out of the <molecule> </molecule> tags the --cot prompt asks for.

    Text without an opening tag is returned unchanged, so runs without --cot are unaffected.
    The *last* tagged block wins: a model that replays the few-shot examples before answering
    puts its own answer last. A generation truncated by --max_completion_length before its
    closing tag falls back to everything after the last opening tag.
    """
    if not isinstance(text, str):
        return text
    blocks = MOLECULE_TAG.findall(text)
    if blocks:
        return blocks[-1].strip()
    if "<molecule>" in text:
        return text.rsplit("<molecule>", 1)[-1].strip()
    return text.strip()

def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol) if mol is not None else None
        return smiles if smiles else None
    except:
        return None

def to_mol(smiles):
    # same filter as canonicalize (unparseable and empty both rejected), but keeps the mol:
    # a SMILES RDKit can read may canonicalize to one it cannot read back
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol is not None and mol.GetNumAtoms() > 0 else None
    except:
        return None

def get_train_molecules(args):
    train_path = f'{args.path}/{args.split}.json'
    if not os.path.exists(train_path):
        return None
    with open(train_path, 'r', encoding="utf8") as f:
        questions = json.load(f)
    if "&" not in args.prop:
        questions = [item for item in questions if item[f'Code_{args.prop}'] != "=0"]

    molecules = set()
    for item in tqdm.tqdm(questions, desc="Building novelty reference set"):
        for key in ["SMILES", "modifiedSMILES"]:
            smiles = canonicalize(item[key])
            if smiles is not None:
                molecules.add(smiles)
    return molecules

def get_diversity(generations):
    fps = []
    for smiles in generations:
        mol = to_mol(smiles)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprint(mol, 2))
    if len(fps) < 2:
        return None

    distances = []
    for i in range(1, len(fps)):
        distances.extend([1 - sim for sim in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    return sum(distances)/len(distances)

def get_sa(preds):
    try:
        import sys
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
    except ImportError as e:
        print(f"SA scorer unavailable ({type(e).__name__}: {e}), skipping SA")
        return None

    sas = []
    for pred in preds:
        mol = to_mol(pred)
        if mol is None:
            sas.append(None)
        else:
            try:
                sas.append(sascorer.calculateScore(mol))
            except Exception:
                sas.append(None)
    return sas if any(sa is not None for sa in sas) else None

def get_scores_generation(eval_outputs, args):
    df = pd.concat([pd.DataFrame(output) for output in eval_outputs])
    # strip the <molecule> </molecule> tags --cot asks for before anything reaches RDKit, so the
    # CSV and every metric below see the SMILES rather than the reasoning around it
    df["pred"] = [extract_molecule(pred) for pred in df["pred"].values.tolist()]
    if "generations" in df.columns:
        # generations are "|"-joined by llm.py; extract per generation and re-join
        df["generations"] = [
            "|".join(extract_molecule(generation) for generation in generations.split("|"))
            for generations in df["generations"].values.tolist()
        ]
    os.makedirs(f'{args.output_dir}/{args.split}/{args.prop}', exist_ok=True)
    csv_path = f'{args.output_dir}/{args.split}/{args.prop}/{args.model_name}_{args.llm_name}_llm_frozen{args.llm_frozen}_{args.split}.csv'
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    validities = []
    accuracies = []
    similarities = []
    for pred, label, code in zip(df["pred"].values.tolist(), df["label"].values.tolist(), df["codes"].values.tolist()):
        try:
            mol_pred, mol_label = Chem.MolFromSmiles(pred), Chem.MolFromSmiles(label)
            validities.append(1)

            hits = hits_check(mol_pred, mol_label, code, args.prop)
            similarities.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(mol_pred, 2), AllChem.GetMorganFingerprint(mol_label, 2)))
            # hits.append(similarities[-1] >= 0.5)
            accuracies.append(all(hits))
        except:
            validities.append(0)

    train_molecules, diversities, sas = None, [], None
    if not args.accuracy_only:
        train_molecules = get_train_molecules(args)
        preds = [canonicalize(pred) for pred in df["pred"].values.tolist()]
        pred_molecules = set([pred for pred in preds if pred is not None])
        if train_molecules:
            seen_molecules = pred_molecules & train_molecules
            df["novel"] = [None if pred is None else pred not in train_molecules for pred in preds]

        if "generations" in df.columns:
            for generations in tqdm.tqdm(df["generations"].values.tolist(), desc="Computing diversity"):
                diversities.append(get_diversity(generations.split("|")))
            df["diversity"] = diversities
        diversities = [diversity for diversity in diversities if diversity is not None]

        sas = get_sa(df["pred"].values.tolist())
        if sas is not None:
            df["sa"] = sas
            sas = [sa for sa in sas if sa is not None]
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    print("Overall")
    print("Validity: {:.2f}% | Accuracy: {:.2f}%".format(
        100*sum(validities)/len(validities), 100*sum(accuracies)/len(validities)
    ))
    if not args.accuracy_only:
        if similarities:
            print("Similarity: {:.2f} (mean pairwise Tanimoto similarity over {}/{} valid predictions)".format(
                sum(similarities)/len(similarities), len(similarities), len(validities)
            ))
        if sas:
            print("SA: {:.2f} (mean over {}/{} valid predictions, lower is easier to synthesize)".format(
                sum(sas)/len(sas), len(sas), len(validities)
            ))
        else:
            print("SA: N/A (requires RDKit's Contrib/SA_Score)")
        if train_molecules:
            print("Novelty: {:.2f}% (|M| = {}, |S| = {}, |M and S| = {})".format(
                100*(1 - len(seen_molecules)/len(train_molecules)), len(pred_molecules), len(train_molecules), len(seen_molecules)
            ))
        else:
            print("Novelty: N/A (training split not found at {}/{}.json)".format(args.path, args.split))
        if diversities:
            print("Diversity: {:.2f}% (mean pairwise Tanimoto distance over {}/{} molecules with >= 2 valid generations)".format(
                100*sum(diversities)/len(diversities), len(diversities), len(validities)
            ))
        else:
            print("Diversity: N/A (--num_return_sequences must be > 1)")

eval_funcs = {
    'generation': get_scores_generation,
}