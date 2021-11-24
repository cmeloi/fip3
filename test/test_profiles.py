import unittest

from pandas import DataFrame

from fip.profiles import *

FEATURE_TUPLES = (('a', 'b', 'c', 'd'), ('a', 'b', 'x'), ('c', 'd'))
COOCCURRENCE_COUNTS = {('a', 'a'): 2, ('a', 'b'): 2, ('a', 'c'): 1, ('a', 'd'): 1, ('a', 'x'): 1,
                       ('b', 'b'): 2, ('b', 'c'): 1, ('b', 'd'): 1, ('b', 'x'): 1,
                       ('c', 'c'): 2, ('c', 'd'): 2, ('d', 'd'): 2, ('x', 'x'): 1}
COOCCURRENCE_PROBABILITIES = {cooccurrence: float(count) / len(FEATURE_TUPLES)
                              for cooccurrence, count in COOCCURRENCE_COUNTS.items()}
COOCCURRENCE_PMI = {cooccurrence: numpy.log2(probability / (
        COOCCURRENCE_PROBABILITIES[(cooccurrence[0], cooccurrence[0])] *
        COOCCURRENCE_PROBABILITIES[(cooccurrence[1], cooccurrence[1])])) if cooccurrence[0] != cooccurrence[1] else 0
                    for cooccurrence, probability in COOCCURRENCE_PROBABILITIES.items()}


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
        cooccurrences = set(CooccurrenceProfile.features2cooccurrences(features))
        self.assertSetEqual(cooccurrences, reference_cooccurrences)

    def test_get_feature_relation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        for feature_tuple, count in COOCCURRENCE_COUNTS.items():
            self.assertEqual(p.interrelation_value(feature_tuple[0], feature_tuple[1]), count)
            self.assertEqual(p.interrelation_value(feature_tuple[1], feature_tuple[0]), count)

    def test_add(self):
        p_cooccurrence_counts = dict(COOCCURRENCE_COUNTS)
        p_cooccurrence_counts.update({('y', 'y'): 1})
        q_cooccurrence_counts = dict(COOCCURRENCE_COUNTS)
        q_cooccurrence_counts.update({('z', 'z'): 2})
        p = CooccurrenceProfile.from_dict(p_cooccurrence_counts, vector_count=4) +\
            CooccurrenceProfile.from_dict(q_cooccurrence_counts, vector_count=5)
        r_cooccurrence_counts = {key: value*2 for key, value in COOCCURRENCE_COUNTS.items()}
        r_cooccurrence_counts.update({('y', 'y'): 1, ('z', 'z'): 2})
        reference_profile = CooccurrenceProfile.from_dict(r_cooccurrence_counts, vector_count=9)
        self.assertEqual(p.attrs['vector_count'], reference_profile.attrs['vector_count'])
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))

    def test_generate_profile_pair_for_feature(self):
        positive_profile, negative_profile = CooccurrenceProfile.from_feature_lists_split_on_feature(
            FEATURE_TUPLES, 'x')
        positive_tuples = [tup for tup in FEATURE_TUPLES if 'x' in tup]
        negative_tuples = [tup for tup in FEATURE_TUPLES if 'x' not in tup]
        self.assertEqual(positive_profile.attrs['vector_count'], len(positive_tuples))
        self.assertEqual(negative_profile.attrs['vector_count'], len(negative_tuples))
        reference_positive_profile = CooccurrenceProfile.from_feature_lists(positive_tuples)
        reference_negative_profile = CooccurrenceProfile.from_feature_lists(negative_tuples)
        self.assertTrue(positive_profile.df.equals(reference_positive_profile.df))
        self.assertTrue(negative_profile.df.equals(reference_negative_profile.df))
        combined_profile = positive_profile + negative_profile
        reference_combined_profile = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        combined_profile.df.sort_index(inplace=True)
        reference_combined_profile.df.sort_index(inplace=True)
        self.assertTrue(combined_profile.df.equals(reference_combined_profile.df))

    def test_distinct_features(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        features = set()
        for feature_tuple in FEATURE_TUPLES:
            features.update(feature_tuple)
        self.assertSetEqual(p.distinct_features(), features)


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
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES), vector_count=len(FEATURE_TUPLES) * 10)
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))


class TestPointwiseMutualInformationProfile(unittest.TestCase):
    def test_pmi_calculation(self):
        reference_profile = PointwiseMutualInformationProfile(DataFrame.from_dict(COOCCURRENCE_PMI,
                                                                                  orient='index', columns=['value']))
        reference_profile.df.sort_index(inplace=True)
        p = PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)))
        p.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))


class TestPointwiseKLDivergenceProfile(unittest.TestCase):
    def test_pointwise_kld_calculation(self):
        cpp1 = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(cpp1, cpp1)
        self.assertTrue((p.df['value'] == 0).all(), "KL divergence with itself should be 0")
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            cpp1,
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])))
        self.assertEqual(p.interrelation_value('a', 'b'), np.log2(2.0 / 3))
        print(p.df)


if __name__ == '__main__':
    unittest.main()
