#!/usr/bin/env python3

from rdkit.Chem import AllChem as Chem


def smiles2rdmol(smiles):
    return Chem.MolFromSmiles(smiles)


def smarts2rdmol(smarts):
    return Chem.MolFromSmarts(smarts)


def rdmol_has_substructure_pattern(rdmol, pattern):
    if not rdmol:
        return None
    return rdmol.HasSubstructMatch(pattern)


def sdf2rdmols(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path)
    for rdmol in supplier:
        if rdmol:
            yield rdmol


def rdmol2fragment_smiles(mol, fragment_locations):
    ecfp_fragment_smiles = set()
    for atom, radius in fragment_locations:
        if radius == 0:
            ecfp_fragment_smiles.add(mol.GetAtomWithIdx(atom).GetSymbol())
        else:
            substructure = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom))
            ecfp_fragment_smiles.add(Chem.MolToSmiles(substructure, canonical=True))
    return ecfp_fragment_smiles


def rdmol2ecfp_features(mol, radius=3):
    bit_info = {}
    features = set()
    Chem.GetMorganFingerprint(mol, radius, bitInfo=bit_info)
    for fragment_id, fragment_locations in bit_info.items():
        features.update(rdmol2fragment_smiles(mol, fragment_locations))
    return features
