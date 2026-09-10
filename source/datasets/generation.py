import json
from torch.utils.data import Dataset
from source.utils.evaluation import *

class GenerationDataset(Dataset):
    def __init__(self, path, prop, split="train", icl=False):
        super().__init__()
        self.prop = prop
        self.icl = icl
        with open(f'{path}/{split}.json', 'r', encoding="utf8") as f:
            self.questions = json.load(f)
        with open(f'{path}/valid.json', 'r', encoding="utf8") as f:
            self.database = json.load(f)

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.questions)

    def __getitem__(self, index):
        item = self.questions[index]

        codes, texts = [], []
        for prop in self.prop.split("&"):
            codes.append(item[f'Code_{prop}']), texts.append(item[f'Text_{prop}'])
        codes_, texts_ = "&".join(codes), " and ".join(texts)

        question = f'Given a molecule, modify it to {texts_}. The modified molecule should be similar to the original molecule.\nRespond with only the SMILES string of your modified molecule. No explanation is needed.'
        if self.icl:
            examples = ""
            for e in item[f'Examples_{prop}'].split(","):
                e = int(e.strip())
                examples += f'\nMolecule:{self.database[e]["SMILES"]}\nAnswer:{self.database[e]["modifiedSMILES"]}'
            question += f'\n\nYou are given a few examples to learn from. Examples:{examples}'

        return {"id": index,
            "smiles": item["SMILES"],
            "prompt": f'{question}\n\nYour task:\nMolecule:{item["SMILES"]}\nAnswer:',
            "label": item["modifiedSMILES"],
            "codes": codes_, "texts": texts_
        }