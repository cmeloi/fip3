import unittest

from fip.chem import *

ASPIRIN_SMILES = "O=C(C)Oc1ccccc1C(=O)O"
ASPIRIN_MORGAN_FRAGMENTS = {
    'cC(=O)O', 'CC(=O)O', 'cc(O)c', 'Cc(c)c(OC(C)=O)cc', 'O=C(O)c1ccccc1O', 'COc1ccccc1C(=O)O',
    'Cc(c)ccc', 'Cc1ccccc1', 'c', 'ccc', 'cc(OC(C)=O)c', 'ccc(C(=O)O)c(O)c', 'ccccc', 'CC(=O)Oc1ccccc1C(=O)O',
    'cOC(C)=O', 'CO', 'COc(cc)c(C)c', 'CC', 'C', 'COc1ccccc1C', 'cccc(O)c', 'Cc(c)c', 'Oc1ccccc1', 'cc(C(=O)O)c',
    'COc', 'O', 'C=O'}

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


class TestFingerprintGenerator(unittest.TestCase):
    def test_fingerprint_generator(self):
        fg = FingerprintGenerator()
        fingerprint1 = fg(ASPIRIN_SMILES+'C', 'morgan', radius=3, fpSize=1024)
        generator1 = fg.generator
        fingerprint2 = fg(ASPIRIN_SMILES, 'morgan', radius=3, fpSize=1024)
        generator2 = fg.generator
        fingerprint3 = fg(ASPIRIN_SMILES, 'rdkit', fpSize=1024)
        generator3 = fg.generator
        fingerprint4 = fg(ASPIRIN_SMILES, 'morgan', radius=3, fpSize=1024)
        self.assertNotEqual(fingerprint1, fingerprint2)
        self.assertNotEqual(fingerprint2, fingerprint3)
        self.assertEqual(fingerprint2, fingerprint4)
        self.assertEqual(generator1, generator2)
        self.assertNotEqual(generator2, generator3)


if __name__ == '__main__':
    unittest.main()
