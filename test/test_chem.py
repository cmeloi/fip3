import unittest

from fip.chem import *

ASPIRIN_SMILES = "O=C(C)Oc1ccccc1C(=O)O"
ASPIRIN_MORGAN_FRAGMENTS = {'[cH][cH][cH]', '[C][O][c]([cH][cH])[c]([C])[cH]', '[cH][cH][cH][cH][c]',
                            '[c][O][C]([CH3])=[O]', '[cH][c]([c])[O]', '[c][cH][cH][cH][cH]',
                            '[O]=[C]([OH])[c]1[cH][cH][cH][cH][c]1[O]', '[C]=[O]', '[c][C](=[O])[OH]',
                            '[cH][cH][c]([C](=[O])[OH])[c]([cH])[O]', '[O][c]1[cH][cH][cH][cH][c]1',
                            '[C][O][c]1[cH][cH][cH][cH][c]1[C]', '[C][CH3]', '[OH]', '[C]', '[O]', '[cH]', '[c]',
                            '[CH3]', '[C][c]([cH])[c]([cH][cH])[O][C]([CH3])=[O]', '[cH][cH][c]',
                            '[CH3][C](=[O])[O][c]1[cH][cH][cH][cH][c]1[C](=[O])[OH]', '[C][c]([c])[cH][cH][cH]',
                            '[C][O][c]', '[cH][cH][cH][c]([c])[O]', '[CH3][C](=[O])[O]', '[C][c]([c])[cH]',
                            '[C][O][c]1[cH][cH][cH][cH][c]1[C](=[O])[OH]', '[C][c]1[c][cH][cH][cH][cH]1',
                            '[c][c]([cH])[C](=[O])[OH]', '[cH][c]([c])[O][C]([CH3])=[O]', '[c][cH][cH]', '[C][OH]'}
ASPIRIN_BRICS = {'[1*]C(C)=O', '[6*]C(=O)O', '[16*]c1ccccc1[16*]', '[3*]O[3*]'}


class TestFragmentGeneration(unittest.TestCase):
    def test_rdmol2morgan_feature_smiles(self):
        mol = smiles2rdmol(ASPIRIN_SMILES)
        feature_smiles_radius_0_to_3 = rdmol2morgan_feature_smiles(mol, radius=3, min_radius=0)
        self.assertSetEqual(feature_smiles_radius_0_to_3, ASPIRIN_MORGAN_FRAGMENTS)
        feature_smiles_radius_1_to_3 = rdmol2morgan_feature_smiles(mol, radius=3, min_radius=1)
        feature_smiles_radius_0_to_0 = rdmol2morgan_feature_smiles(mol, radius=0, min_radius=0)
        for smile in feature_smiles_radius_0_to_0:
            self.assertNotIn(smile, feature_smiles_radius_1_to_3)
        feature_smiles_radius_1_to_3.update(feature_smiles_radius_0_to_0)
        self.assertSetEqual(feature_smiles_radius_1_to_3, feature_smiles_radius_0_to_3)

    def test_rdmol2morgan_feature_smiles_too_small(self):
        mol = smiles2rdmol('CCO')  # too small for radius 2
        feature_smiles_radius_2_to_2 = rdmol2morgan_feature_smiles(mol, radius=2, min_radius=2)
        self.assertSetEqual(feature_smiles_radius_2_to_2, set())

    def test_rdmol2brics_blocs_smiles(self):
        mol = smiles2rdmol(ASPIRIN_SMILES)
        brics_blocs = rdmol2brics_blocs_smiles(mol)
        self.assertSetEqual(brics_blocs, ASPIRIN_BRICS)

    def test_rdmol2smiles(self):
        mol1 = smiles2rdmol(ASPIRIN_SMILES)
        mol2 = smiles2rdmol(Chem.MolFromSmiles(ASPIRIN_SMILES))
        self.assertEqual(rdmol2smiles(mol1), rdmol2smiles(mol2))


if __name__ == '__main__':
    unittest.main()
