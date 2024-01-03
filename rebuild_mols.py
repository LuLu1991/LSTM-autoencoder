from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds


dict_brics ={
    "[1*]":["[2*]","[3*]","[10*]"],
    "[2*]":["[1*]","[14*]","[12*]","[16*]"],
    "[3*]":["[1*]","[4*]","[13*]","[14*]","[15*]","[16*]"],
    "[4*]":["[3*]","[5*]","[11*]"],
    "[5*]":["[4*]","[5*]","[15*]"],
    "[6*]":["[8*]","[13*]","[15*]","[16*]"],
    "[7*]":["[7*]"],
    "[8*]":["[9*]","[10*]","[13*]","[14*]","[15*]","[16*]"],
    "[9*]":["[14*]","[15*]","[16*]"],
    "[10*]":["[1*]","[8*]","[13*]","[14*]","[15*]","[16*]"],
    "[11*]":["[4*]","[13*]","[14*]","[15*]","[16*]"],
    "[12*]":["[2*]"],
    "[13*]":["[3*]","[5*]","[6*]","[8*]","[10*]","[11*]","[14*]","[15*],""[16*]"],
    "[14*]":["[2*]","[3*]","[6*]","[8*]","[10*]","[11*]","[13*]","[15*],""[16*]"],
    "[15*]":["[3*]","[5*]","[6*]","[8*]","[9*]","[10*]","[11*]","[13*]","[14*]","[16*]"],
    "[16*]":["[2*]","[3*]","[6*]","[8*]","[9*]","[10*]","[11*]","[13*]","[14*]","[15*]"],
}



def get_repl_idx(combo,smarts):
    for atom in combo.GetAtoms():
        if atom.GetSmarts() == smarts:
            repl_idx = atom.GetIdx()
            break
    return repl_idx

def get_comb_idx(combo,smarts):
    comb_idx_list = []
    for atom in combo.GetAtoms():
        if atom.GetSmarts() == smarts:
            neighbors = atom.GetNeighbors()
            comb_idx = neighbors[0].GetIdx()
            comb_idx_list.append(comb_idx)
    return comb_idx_list

def get_smarts(frags):
    smarts_list=[]
    for atom in frags.GetAtoms():
        if "*" in atom.GetSmarts():
            smarts_list.append(atom.GetSmarts())
    return smarts_list

def get_idx_smarts(frags):
    idx_smarts = dict()
    for atom in frags.GetAtoms():
        if "*" in atom.GetSmarts():
            idx_smarts[atom.GetIdx()] = atom.GetSmarts()
    return idx_smarts

def get_id_pairs(frags_0_idx_smarts,frags_1_idx_smarts):
    pro_id_pairs = []
    mis_id_pairs = []
    for x in frags_0_idx_smarts:
        smarts_0 = frags_0_idx_smarts[x]
        for y in frags_1_idx_smarts:
            smarts_1 = frags_1_idx_smarts[y]
            if smarts_1 in dict_brics[smarts_0]:
                pro_id_pairs.append([x , y])
            else:
                mis_id_pairs.append([x,y])
    return pro_id_pairs,mis_id_pairs

def repl_atom(mols,id_pairs):
    edmol_0 = Chem.EditableMol(mols[0])
    edmol_0.ReplaceAtom(id_pairs[0][0],Chem.AtomFromSmarts("*"))
    back_0 = edmol_0.GetMol()
    edmol_1 = Chem.EditableMol(mols[1])
    edmol_1.ReplaceAtom(id_pairs[0][1],Chem.AtomFromSmarts("*"))
    back_1 = edmol_1.GetMol()
    return back_0,back_1

def combo_two_frags(back_0,back_1):
    combo = Chem.CombineMols(back_0,back_1)
    combo_id_list = get_comb_idx(combo,"*")
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(combo_id_list[0],combo_id_list[1],order=Chem.rdchem.BondType.SINGLE)
    back = edcombo.GetMol()
    for i in range(2):
        repl_idx=get_repl_idx(back,"*")
        edcombo = Chem.EditableMol(back)
        edcombo.RemoveAtom(repl_idx)
        back = edcombo.GetMol()
    return back



def convert_frags_to_smiles(frags):
    idx_list =[]
    for i,smi in enumerate(frags):
        if smi.count("*") == 1:
            idx_list.append(i)
    idx_start = 0
    slice_list = []
    if frags[0].count("*") == 1:
        for i in idx_list:
            temp_list = [idx_start,i]
            idx_start = i
            slice_list.append(temp_list)
        slice_list.append([idx_start,None])
        #print("1")
    else:
        for i in idx_list:
            temp_list = [idx_start,i+1]
            idx_start = i+1
            slice_list.append(temp_list)
        slice_list.append([idx_start,None])
        #print("2")
    new_frags = []
    for i in slice_list:
        new_frags.append(frags[i[0]:i[1]])
    if [] in new_frags:
        new_frags.remove([])
    new_frags_mols = []
    for i in new_frags:
        new_frags_mols.append([Chem.MolFromSmiles(smi) for smi in i])
    pre_combo_list = []
    for x in new_frags_mols:
        if len(x) == 1:
            pre_combo_list.append(x[0])
        else:
            pre_combo = x[0]
            for i in x[1:]:
                frags_0_idx_smarts = get_idx_smarts(pre_combo)
                frags_1_idx_smarts = get_idx_smarts(i)
                pro_id_pairs,mis_id_pairs = get_id_pairs(frags_0_idx_smarts,frags_1_idx_smarts)
                mols_pair =[pre_combo,i]
                if len(pro_id_pairs) != 0:
                    back_0,back_1 = repl_atom(mols_pair,pro_id_pairs)
                else:
                    back_0,back_1 = repl_atom(mols_pair,mis_id_pairs)
                pre_combo = combo_two_frags(back_0,back_1)
            pre_combo_list.append(pre_combo)
    if len(pre_combo_list) == 1:
        combo_mol = pre_combo_list[0]
    else:
        combo_mol = pre_combo_list[0]
        for i in pre_combo_list[1:]:
            frags_0_idx_smarts = get_idx_smarts(combo_mol)
            frags_1_idx_smarts = get_idx_smarts(i)
            pro_id_pairs,mis_id_pairs = get_id_pairs(frags_0_idx_smarts,frags_1_idx_smarts)
            mols_pair =[combo_mol,i]
            if len(pro_id_pairs) != 0:
                back_0,back_1 = repl_atom(mols_pair,pro_id_pairs)
            else:
                back_0,back_1 = repl_atom(mols_pair,mis_id_pairs)
            combo_mol = combo_two_frags(back_0,back_1)
    pre_del_idx = []
    for atom in combo_mol.GetAtoms():
        if "*" in atom.GetSmarts():
            pre_del_idx.append(atom.GetIdx())
    #print(len(pre_del_idx))
    if pre_del_idx == []:
        combo_mol = combo_mol
    else:
        for i in range(len(pre_del_idx)):
            for atom in combo_mol.GetAtoms():
                while "*" in atom.GetSmarts():
                    del_idx = atom.GetIdx()
                    #print(del_idx)
                    break
            #print(del_idx)
            edcombo_mol = Chem.EditableMol(combo_mol)
            edcombo_mol.RemoveAtom(del_idx)
            combo_mol = edcombo_mol.GetMol()
    out_smiles = Chem.MolToSmiles(combo_mol)
    return out_smiles
