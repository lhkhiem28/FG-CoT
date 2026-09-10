import json
from torch.utils.data import Dataset
from source.utils.evaluation import *

class GenerationDataset(Dataset):
    def __init__(self, path, prop, split="train", icl=False, cot=False, lite=False):
        super().__init__()
        self.prop = prop
        self.icl = icl
        self.cot = cot
        self.lite = lite
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

        if not self.cot:
            question = f'Given a molecule, modify it to {texts_}. The modified molecule should be similar to the original molecule.\nRespond with only the SMILES string of your modified molecule. No explanation is needed.'
        else:
            if not self.lite:
                pass
            else:
                question = f'Given a molecule, modify it to {texts_}. The modified molecule should be similar to the original molecule.\nPlease reason about which functional groups to remove and/or add, and at which positions on the molecule, before modifying it. Put your modified molecule within <molecule> </molecule> tags.'

        if self.icl:
            examples = ""
            for e in item[f'Examples_{self.prop}'].split(","):
                e = int(e.strip())
                if not self.cot:
                    examples += f'\nMolecule:{self.database[e]["SMILES"]}\nAnswer:{self.database[e]["modifiedSMILES"]}'
                else:
                    if not self.lite:
                        examples += f'\nMolecule:{self.database[e]["SMILES"]}\nReasoning:\n{self.database[e]["CoT"]}\nThe modified molecule is: <molecule> {self.database[e]["modifiedSMILES"]} </molecule>'
                    else:
                        examples += f'\nMolecule:{self.database[e]["SMILES"]}\nReasoning:\n{self.database[e]["CoT-Lite"]}\nThe modified molecule is: <molecule> {self.database[e]["modifiedSMILES"]} </molecule>'
            question += f'\n\nYou are given a few examples to learn from. Examples:{examples}'

        return {"id": index,
            "smiles": item["SMILES"],
            "prompt": f'{question}\n\nYour task:\nMolecule:{item["SMILES"]}\nAnswer:' if not self.cot else f'{question}\n\nYour task:\nMolecule:{item["SMILES"]}',
            "label": item["modifiedSMILES"],
            "codes": codes_, "texts": texts_
        }