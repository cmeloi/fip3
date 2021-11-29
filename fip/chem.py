#!/usr/bin/env python3

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.BRICS import BRICSDecompose


def smiles2rdmol(smiles):
    """Simple conversion of SMILES string into a RDKit Mol instance.
    Wrapped in case some standardization/postprocessing needed.

    :param smiles: SMILES string
    :return: RDKit Mol instance
    """
    return Chem.MolFromSmiles(smiles)


def smarts2rdmol(smarts):
    """Simple conversion of SMARTS string into a RDKit Mol instance.
    Wrapped in case some standardization/postprocessing needed.

    :param smarts: SMARTS string
    :return: RDKit Mol instance
    """
    return Chem.MolFromSmarts(smarts)


def rdmol_has_substructure_pattern(rdmol, pattern):
    """Direct wrap of RDKit HasSubstructMatch.
    Wrapped in case some standardization/postprocessing needed.

    :param rdmol: RDKit Mol instance to search the pattern in
    :param pattern: the pattern to search, also in RDKit Mol form
    :return: bool or None (in case of RDKit Mol instance error)
    """
    if not rdmol:
        return None
    return rdmol.HasSubstructMatch(pattern)


def sdf2rdmols(sdf_path):
    """Converts an SDF file to RDKit Mol instances.
    Wrap of RDKit SDMolSupplier, modified to omit RDKit Mol instances that can't be parsed.

    :param sdf_path: Path to the SDF file
    :return: a generator yielding RDKit Mol instances
    """
    supplier = Chem.SDMolSupplier(sdf_path)
    for rdmol in supplier:
        if rdmol:
            yield rdmol


def rdmol2fragment_smiles(mol, fragment_locations, min_radius=0):
    """Generates a set of fragments in SMILES notation from a molecule given as an RDKit Mol instance,
    based on provided atom indices and radii.

    :param mol: the molecule for fragmenting, as RDKit Mol
    :param fragment_locations: an iterable of fragment locations, in (atom, radius) tuples
    :param min_radius: minimal feature radius to consider, default 0. Can be set to ignore lower-scope features.
    :return: a set of SMILES strings
    """
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
            fragment_smiles.add(Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms),
                                                         bondsToUse=bonds, canonical=True))
    return fragment_smiles


def rdmol2morgan_feature_smiles(mol, radius=3, min_radius=1):
    """Breaks a molecule, given as an RDKit Mol instance, into a set of ECFP-like fragments with selected radius, and returns them in SMILES notation.

    :param mol: the molecule for fragmenting, as RDKit Mol
    :param radius: EC fragment radius
    :param min_radius: minimal fragment radius to consider, default 0. Can be set to ignore lower-scope fragments.
    :return: a set of SMILES strings
    """
    bit_info = {}
    features = set()
    Chem.GetMorganFingerprint(mol, radius, bitInfo=bit_info)
    for fragment_id, fragment_locations in bit_info.items():
        features.update(rdmol2fragment_smiles(mol, fragment_locations, min_radius=min_radius))
    return features


def rdmol2brics_blocs_smiles(mol, min_fragment_size=1):
    """Decomposes the provided rdmol instance into BRICS fragment SMILES, as designed by Degen et al. in
    On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces, https://doi.org/10.1002/cmdc.200800178

    :param mol: the source RDKit Mol instance
    :param min_fragment_size: passed to the minFragmentSize of the RDKit implementation
    :return: SMILES of the BRICS fragments
    """
    return BRICSDecompose(mol, minFragmentSize=min_fragment_size, returnMols=False)
