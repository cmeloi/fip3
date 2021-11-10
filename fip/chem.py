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


def rdmol2fragment_smiles(mol, fragment_locations, min_radius=0):
    fragment_smiles = set()
    for atom, radius in fragment_locations:
        if radius < min_radius:
            continue
        if radius == 0:
            fragment_smiles.add(mol.GetAtomWithIdx(atom).GetSymbol())
        else:
            atoms = set()
            bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
            for bond_id in bonds:
                bond = mol.GetBondWithIdx(bond_id)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            fragment_smiles.add(Chem.MolFragmentToSmiles(mol,
                                                         atomsToUse=list(atoms),
                                                         bondsToUse=bonds,
                                                         canonical=True))
            #substructure = Chem.PathToSubmol(mol, Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom))
            #ecfp_fragment_smiles.add(Chem.MolToSmiles(substructure, canonical=True))
    return fragment_smiles


def rdmol2morgan_feature_smiles(mol, radius=3, min_radius=1):
    bit_info = {}
    features = set()
    Chem.GetMorganFingerprint(mol, radius, bitInfo=bit_info)
    for fragment_id, fragment_locations in bit_info.items():
        features.update(rdmol2fragment_smiles(mol, fragment_locations, min_radius=min_radius))
    return features
