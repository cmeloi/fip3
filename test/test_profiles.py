import unittest

from pandas import DataFrame

from fip.profiles import CooccurrenceProfile, CooccurrenceProbabilityProfile


FEATURE_TUPLES = (('a', 'b', 'c', 'd'), ('a', 'b', 'x'), ('c', 'd'))
COOCCURRENCE_COUNTS = {('a', 'a'): 2, ('a', 'b'): 2, ('a', 'c'): 1, ('a', 'd'): 1, ('a', 'x'): 1,
                       ('b', 'b'): 2, ('b', 'c'): 1, ('b', 'd'): 1, ('b', 'x'): 1,
                       ('c', 'c'): 2, ('c', 'd'): 2, ('d', 'd'): 2, ('x', 'x'): 1}
COOCCURRENCE_PROBABILITIES = {cooccurrence: float(count)/len(FEATURE_TUPLES)
                              for cooccurrence, count in COOCCURRENCE_COUNTS.items()}


class TestCooccurrenceProfile(unittest.TestCase):
    def test_cooccurrence_counting(self):
        reference_profile = CooccurrenceProfile(DataFrame.from_dict(COOCCURRENCE_COUNTS,
                                                                    orient='index', columns=['value']))
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))

    def test_features2cooccurrences(self):
        features = ('c', 'a', 'b', 'a')
        reference_cooccurrences = {('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')}
        cooccurrences = set(CooccurrenceProfile._features2cooccurrences(features))
        self.assertSetEqual(cooccurrences, reference_cooccurrences)

    def test_get_feature_relation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        for feature_tuple, count in COOCCURRENCE_COUNTS.items():
            self.assertEqual(p.get_feature_relation(feature_tuple[0], feature_tuple[1]), count)
            self.assertEqual(p.get_feature_relation(feature_tuple[1], feature_tuple[0]), count)


class TestCooccurrenceProbabilityProfile(unittest.TestCase):
    def test_cooccurrence_probability_calculation(self):
        reference_profile = CooccurrenceProbabilityProfile(DataFrame.from_dict(COOCCURRENCE_PROBABILITIES,
                                                                               orient='index', columns=['value']))
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))

    def test_cooccurrence_probability_calculation_explicit_count(self):
        reference_profile = CooccurrenceProbabilityProfile(DataFrame.from_dict(COOCCURRENCE_PROBABILITIES,
                                                                               orient='index', columns=['value']))
        reference_profile.df = reference_profile.df.divide(10)
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES), vector_count=len(FEATURE_TUPLES)*10)
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))


if __name__ == '__main__':
    unittest.main()
