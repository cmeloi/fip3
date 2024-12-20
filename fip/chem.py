#!/usr/bin/env python3

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.rdmolops import RemoveStereochemistry, RemoveAllHs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import rdFingerprintGenerator


class FingerprintGenerator:
    """A wrapper for RDKit fingerprint generators, allowing for easy generation of fingerprints with different settings.
    Can instantiate a generator ad-hoc for a given type and settings, and then generate fingerprints for molecules.
    The instantieted generator is stored and reused for the same type and settings,
    to avoid unnecessary reinitialization.

    Supported fingerprint types:
    - 'rdkit': RDKit fingerprint
    - 'atom_pair': Atom pair fingerprint
    - 'morgan': Morgan fingerprint
    - 'topological_torsion': Topological torsion fingerprint

    Supported generator settings:
    - 'rdkit': ['fpSize']
    - 'atom_pair': ['fpSize']
    - 'morgan': ['radius', 'fpSize', 'countSimulation', 'includeChirality', 'useBondTypes',
                 'onlyNonzeroInvariants', 'includeRingMembership', 'countBound']
    - 'topological_torsion': ['fpSize']

    Supported fingerprint generation settings:
    - ['ignoreAtoms', 'confId', 'customAtomInvariants', 'customBondInvariants', 'additionalOutput']

    Usage:
    generator = FingerprintGenerator()
    fingerprint = generator(mol, type='morgan', radius=3, fpSize=1024)

    """

    generator_type2getter = {
        'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator,
        'atom_pair': rdFingerprintGenerator.GetAtomPairGenerator,
        'morgan': rdFingerprintGenerator.GetMorganGenerator,
        'topological_torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator,
    }
    generator_type2args = {
        'rdkit': ['fpSize'],
        'atom_pair': ['fpSize'],
        'morgan': ['radius', 'fpSize', 'countSimulation', 'includeChirality', 'useBondTypes',
                   'onlyNonzeroInvariants', 'includeRingMembership', 'countBound'],
        'topological_torsion': ['fpSize'],
    }
    get_fingerprint_args = ['ignoreAtoms', 'confId', 'customAtomInvariants', 'customBondInvariants', 'additionalOutput']

    def __init__(self):
        self.generator = None
        self.generator_type = None
        self.generator_args = None

    def __call__(self, mol, type='morgan', **kwargs):
        if isinstance(mol, str):
            mol = smiles2rdmol(mol)
        generator_init_args = {arg: kwargs[arg] for arg in self.generator_type2args[type] if arg in kwargs}
        if not self.generator or self.generator_type != type or self.generator_args != generator_init_args:
            self.generator = self.generator_type2getter[type](**generator_init_args)
            self.generator_type = type
            self.generator_args = kwargs
        return self.generator.GetFingerprint(
            mol, **{arg: kwargs[arg] for arg in self.get_fingerprint_args if arg in kwargs})


FP_GENERATOR = FingerprintGenerator()


def smiles2rdmol(smiles):
    """Simple conversion of SMILES string into a RDKit Mol instance.
    Wrapped in case some standardization/postprocessing needed.
    If the argument is already a RDKit Mol instance, passes it through without change.

    :param smiles: SMILES string or an RDKit Mol instance
    :return: RDKit Mol instance
    """
    if isinstance(smiles, Chem.rdchem.Mol):
        return smiles
    return Chem.MolFromSmiles(smiles)


def smarts2rdmol(smarts):
    """Simple conversion of SMARTS string into a RDKit Mol instance.
    Wrapped in case some standardization/postprocessing needed.
    If the argument is already a RDKit Mol instance, passes it through without change.

    :param smiles: SMARTS string or an RDKit Mol instance
    :return: RDKit Mol instance
    """
    if isinstance(smarts, Chem.rdchem.Mol):
        return smarts
    return Chem.MolFromSmarts(smiles)


def rdmol2smiles(mol):
    """Simple conversion of RDKit Mol instance into a SMILES string.
    Wrapped in case some standardization/postprocessing needed.

    :param mol: RDKit Mol instance
    :return: SMILES string
    """
    return Chem.MolToSmiles(mol)


def smarts2rdmol(smarts):
    """Simple conversion of SMARTS string into a RDKit Mol instance.
    Wrapped in case some standardization/postprocessing needed.

    :param smarts: SMARTS string
    :return: RDKit Mol instance
    """
    return Chem.MolFromSmarts(smarts)


def standardize_mol(mol, *, remove_hydrogens=True, remove_stereo=True):
    """Simple structure standardization.

    :param mol: RDKit Mol instance to standardize
    :param remove_hydrogens: whether to remove all hydrogens using RDKit RemoveAllHs
    :param remove_stereo: whether to remove stereo information using RDKIt RemoveStereochemistry
    :return: standardized RDKit Mol instance
    """
    if remove_stereo:
        RemoveStereochemistry(mol)
    if remove_hydrogens:
        mol = RemoveAllHs(mol, sanitize=True)
    return mol


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


def rdmol_bonds2fragment_smiles(mol, bonds, *, all_bonds_explicit=False, canonical_smiles=True,
                                isomeric_smiles=False, all_H_explicit=True):
    """Generates a fragment in SMILES notation from a molecule given as an RDKit Mol instance,
    based on the provided bond IDs within the molecule.

    :param mol: the molecule containing the fragment, as RDKit Mol
    :param bonds: a set of bonds that are part of the fragment
    :param all_bonds_explicit: boolean indicating whether all bond orders will be explicitly stated in the output. Default False.
    :param canonical_smiles: boolean indicating whether the fragment should be attempted to make canonical. Default True.
    :param isomeric_smiles: boolean indicating whether to include stereo information in the fragments. Default False.
    :param all_H_explicit: boolean indicating whether to explicitly include all hydrogen atoms. Default False.
    :return: a SMILES string of the fragment, None if there are no atoms matched
    """
    atoms = set()
    for bond_id in bonds:
        bond = mol.GetBondWithIdx(bond_id)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    if not atoms:
        return None
    return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=bonds,
                             allBondsExplicit=all_bonds_explicit, canonical=canonical_smiles,
                             isomericSmiles=isomeric_smiles, allHsExplicit=all_H_explicit)


def rdmol_locations2fragments_smiles(mol, fragment_locations, min_radius=0, *, all_bonds_explicit=False,
                                     canonical_smiles=True, isomeric_smiles=False, all_H_explicit=True):
    """Generates a set of fragments in SMILES notation from a molecule given as an RDKit Mol instance,
    based on provided atom indices and radii.

    :param mol: the molecule for fragmenting, as RDKit Mol
    :param fragment_locations: an iterable of fragment locations, in (atom, radius) tuples
    :param min_radius: minimal feature radius to consider, default 0. Can be set to ignore lower-scope features.
    :param all_bonds_explicit: boolean indicating whether all bond orders will be explicitly stated in the output. Default False.
    :param canonical_smiles: boolean indicating whether the fragment should be attempted to make canonical. Default True.
    :param isomeric_smiles: boolean indicating whether to include stereo information in the fragments. Default False.
    :param all_H_explicit: boolean indicating whether to explicitly include all hydrogen atoms. Default True.
    :return: a set of SMILES strings
    """
    fragment_smiles = set()
    for atom, radius in fragment_locations:
        if radius < min_radius:
            continue
        if radius == 0:
            fragment_smiles.add(Chem.MolFragmentToSmiles(mol, atomsToUse=[atom], allBondsExplicit=all_bonds_explicit,
                                                         canonical=canonical_smiles,
                                                         isomericSmiles=isomeric_smiles, allHsExplicit=all_H_explicit))
        else:
            bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom)
            fragment_smiles.add(rdmol_bonds2fragment_smiles(mol, bonds, all_bonds_explicit=all_bonds_explicit,
                                                            canonical_smiles=canonical_smiles,
                                                            isomeric_smiles=isomeric_smiles,
                                                            all_H_explicit=all_H_explicit))
    fragment_smiles.discard(None)  # in case there were some invalid fragments
    return fragment_smiles


def rdmol2morgan_feature_smiles(mol, radius=3, min_radius=1, *, all_bonds_explicit=False,
                                     canonical_smiles=True, isomeric_smiles=False, all_H_explicit=False):
    """Breaks a molecule, given as an RDKit Mol instance, into a set of ECFP-like fragments with selected radius, and returns them in SMILES notation.

    :param mol: the molecule for fragmenting, either as RDKit Mol, or a SMILES string
    :param radius: EC fragment radius
    :param min_radius: minimal fragment radius to consider, default 0. Can be set to ignore lower-scope fragments.
    :param all_bonds_explicit: boolean indicating whether all bond orders will be explicitly stated in the output. Default False.
    :param canonical_smiles: boolean indicating whether the fragment should be attempted to make canonical. Default True.
    :param isomeric_smiles: boolean indicating whether to include stereo information in the fragments. Default False.
    :param all_H_explicit: boolean indicating whether to explicitly include all hydrogen atoms. Default False.
    :return: a set of SMILES strings
    """
    features = set()
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateAtomCounts()
    ao.AllocateAtomToBits()
    ao.AllocateBitInfoMap()
    FP_GENERATOR(mol, 'morgan', radius=radius, fpSize=1024, additionalOutput=ao)
    bit_info = ao.GetBitInfoMap()
    for fragment_id, fragment_locations in bit_info.items():
        features.update(rdmol_locations2fragments_smiles(mol, fragment_locations, min_radius=min_radius,
                                                         all_bonds_explicit=all_bonds_explicit,
                                                         canonical_smiles=canonical_smiles,
                                                         isomeric_smiles=isomeric_smiles,
                                                         all_H_explicit=all_H_explicit))
    return features


def rdmol2brics_blocs_smiles(mol, min_fragment_size=1):
    """Decomposes the provided rdmol instance into BRICS fragment SMILES, as designed by Degen et al. in
    On the Art of Compiling and Using 'Drug-Like' Chemical Fragment Spaces, https://doi.org/10.1002/cmdc.200800178

    :param mol: the source RDKit Mol instance
    :param min_fragment_size: passed to the minFragmentSize of the RDKit implementation
    :return: SMILES of the BRICS fragments
    """
    return BRICSDecompose(mol, minFragmentSize=min_fragment_size, returnMols=False)


def flatten_fragment_contributions(fragment_contributions):
    """Takes an iterable of fragment pairs or larger sets and their contributions, such as (((f1, f2), score), ...),
    {(f1, f2, f3): score, ...} etc. and converts them into individual (fragment, score) pairs. For example,
    (((f1, f2), score1), ((f2, f3), score2)) would be converted into
    ((f1, score1), (f2, score1), (f2, score2), (f3, score2)).

    :param fragment_contributions: iterable of fragment pairs or larger sets and their contributions to flatten
    :return: individual (fragment, score) tuples
    """
    if isinstance(fragment_contributions, dict):
        contribution_iterator = iter(fragment_contributions.items())
    else:
        contribution_iterator = iter(fragment_contributions)
    flattened_contributions = {}
    for fragments, contribution in contribution_iterator:
        if isinstance(fragments, str):
            flattened_contributions[fragments] = flattened_contributions.get(fragments, 0) + contribution
        else:
            for fragment in fragments:
                flattened_contributions[fragment] = flattened_contributions.get(fragment, 0) + contribution
    return flattened_contributions


def rdmol_merge_fragment_contributions(mol, fragment_contributions, *, average=True):
    """Merges individual contributions of given patterns on atoms in a given molecule.

    :param mol: the RDKit Mol instance to apply the contributions to
    :param fragment_contributions: fragments mapping onto their contributions, e.g. {<RDKit Mol | SMILES pattern>: score}
    :param average: Averages the fragment contributions on individual atoms, instead of just summing. Default True.
    :return: a list of merged contributions, indexed to individual atoms in the given molecule
    """
    fragment_contributions = {smarts2rdmol(smiles): score for smiles, score
                              in flatten_fragment_contributions(fragment_contributions).items()}
    atom_values = [0 for atom in mol.GetAtoms()]
    atom_hits = [0 for atom in mol.GetAtoms()]
    for query, contribution in fragment_contributions.items():
        for hit_atoms in mol.GetSubstructMatches(query):
            for hit_atom_id in hit_atoms:
                atom_values[hit_atom_id] += contribution
                atom_hits[hit_atom_id] += 1
    if average:
        atom_values = [value/hit_count if hit_count > 0 else 0 for value, hit_count in zip(atom_values, atom_hits)]
    return atom_values


def rdmol_visualize_fragment_contributions(mol, fragment_contributions, *, drawing_sizes=(400, 400),
                                           average=True, similarity_map_kwargs={}):
    """Merges individual contributions of given patterns on atoms in a given molecule, and visualizes the contributions
    on the molecule in a SVG format.

    :param mol: the RDKit Mol instance to apply the contributions to
    :param fragment_contributions: fragments mapping onto their contributions: {<RDKit Mol | SMILES pattern>: score}
    :param drawing_sizes: a tuple containing the declared sizes of the SVG: (size_x, size_y)
    :param average: Averages the fragment contributions on individual atoms, instead of just summing. Default True.
    :param similarity_map_kwargs: dict of keyword arguments, to be passed to the rdkit GetSimilarityMapFromWeights
    :return: an SVG text with the picture of the molecule
    """
    merged_contributions = rdmol_merge_fragment_contributions(mol, fragment_contributions, average=average)
    drawing = Draw.MolDraw2DSVG(drawing_sizes[0], drawing_sizes[1])
    SimilarityMaps.GetSimilarityMapFromWeights(mol, merged_contributions, draw2d=drawing, **similarity_map_kwargs)
    drawing.FinishDrawing()
    return drawing.GetDrawingText()
