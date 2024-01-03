from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds
import pandas as pd 
import random
import os,sys
import random
import json


def split_smi_to_frags(smi):
    try:
        mol=Chem.MolFromSmiles(smi)
        mol_frags=BreakBRICSBonds(mol)
        mol_frags = Chem.GetMolFrags(mol_frags,asMols=True)
        smi_frags=[Chem.MolToSmiles(x,isomericSmiles=True, canonical=True) for x in mol_frags]
        return smi_frags
    except:
        print(f"Invalid smiles: {smi}")
        return [None]
    #mol_frags = Chem.GetMolFrags(mol_frags,asMols=True)
    #smi_frags=[Chem.MolToSmiles(x,True) for x in mol_frags]

def save_to_json(df, f_name):
    data_dir = "data"
    try:
        os.mkdir(json_dir)
    except:
        print(f"{data_dir} had been built ")
    f_name = f_name
    json_files = os.path.join(data_dir,f_name)
    df.to_json(f"{json_files}.json")
    print(f"{f_name}.json was saved")



def make_new_smiles_frags_vocab(smiles_text):
    new_smiles_text =[]
    new_frags_list = []
    set_vocab = set()
    for smi in smiles_text:
        smi_frags = split_smi_to_frags(smi)
        new_frags = []
        for frag in smi_frags:
            while type(frag) == str:
                if "*"  in frag:
                    new_frags.append(frag)
                    set_vocab.update([frag])
                    temp_smi =smi
                break
        if smi ==temp_smi:
            new_smiles_text.append(smi)
        if len(new_frags) > 0:
            new_frags_list.append(new_frags)
    dict_smiles={"smiles":new_smiles_text,"frags":new_frags_list}
    return set_vocab,dict_smiles

if __name__ == "__main__":
    smiles_file_path = os.path.join("./zinc20_data/zinc_smiles.smi")
    smiles_test_path = os.path.join("./test_data/test_data.csv")
    df_train = pd.read_csv(smiles_file_path,sep=",")
    df_test = pd.read_csv(smiles_test_path,sep=",")
    #df = pd.read_csv(smiles_file_path, delim_whitespace=True, names=["smiles","zinc_name"])
    smiles_text = [x for x in df_train["smiles"]]
    for i in range(5):
        random.shuffle(smiles_text)
    set_vocab,dict_smiles=make_new_smiles_frags_vocab(smiles_text)
    

    df = pd.DataFrame.from_dict(dict_smiles)
    num_val_samples = int(0.25 * len(df))
    num_train_samples = len(df) - num_val_samples
    train_samples = df[:num_train_samples]
    val_samples = df[num_train_samples:]


    save_to_json(train_samples,f_name = "train_data")
    save_to_json(val_samples,f_name = "val_data")

    smiles_text = [x for x in df_test["smiles"]]
    for i in range(5):
        random.shuffle(smiles_text)
    test_vocab,dict_smiles=make_new_smiles_frags_vocab(smiles_text)
    
    df = pd.DataFrame.from_dict(dict_smiles)
    save_to_json(df,f_name = "test_data")
    
    for i in test_vocab:
        set_vocab.update([i])
    
    list_vocab = list(set_vocab)
    list_vocab.sort(key=lambda i:len(i))
    dict_vocab = {"<pad>":0,"<sos>":1,"<eos>":2,"<unk>":3}
    for i, vocab in enumerate(list_vocab):
        key = vocab
        value = int(i+4)
        dict_vocab[key] = value

    try:
        os.mkdir("JSON")
    except:
        print(f"JSON_DIR had been built ")

    with open ("JSON/dict_vocab.json",'w') as f:
        json.dump(dict_vocab,f)
        print("dict_vocab.json was saved")


