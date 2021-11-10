import unittest

from fip.chem import *

ASPIRIN_SMILES = "O=C(C)Oc1ccccc1C(=O)O"
ASPIRIN_MORGAN_FRAGMENTS = {'ccc', 'ccc(C(=O)O)c(c)O', 'Cc1ccccc1', 'ccccc', 'COc(cc)c(C)c', 'cccc(c)O', 'Oc1ccccc1',
                            'Cc(c)ccc', 'O', 'Cc(c)c(cc)OC(C)=O', 'COc1ccccc1C', 'Cc(c)c', 'cOC(C)=O',
                            'CC(=O)Oc1ccccc1C(=O)O', 'COc', 'cc(c)O', 'O=C(O)c1ccccc1O', 'cc(c)OC(C)=O', 'cC(=O)O',
                            'COc1ccccc1C(=O)O', 'CC', 'C', 'CC(=O)O', 'cc(c)C(=O)O', 'C=O', 'CO'}


class TestFragmentGeneration(unittest.TestCase):
    def test_rdmol2morgan_feature_smiles(self):
        mol = smiles2rdmol(ASPIRIN_SMILES)
        feature_smiles_radius_0_to_3 = rdmol2morgan_feature_smiles(mol, radius=3, min_radius=0)
        self.assertSetEqual(feature_smiles_radius_0_to_3, ASPIRIN_MORGAN_FRAGMENTS)
        feature_smiles_radius_1_to_3 = rdmol2morgan_feature_smiles(mol, radius=3, min_radius=1)
        self.assertSetEqual(feature_smiles_radius_1_to_3, {x for x in ASPIRIN_MORGAN_FRAGMENTS if len(x) > 1})


if __name__ == '__main__':
    unittest.main()
