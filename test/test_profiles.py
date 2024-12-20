import statistics
import unittest
from io import StringIO
from pyexpat import features

import numpy as np
from pandas import DataFrame

from fip.profiles import *

FEATURE_TUPLES = (('a', 'b', 'c', 'd'), ('a', 'b', 'x'), ('c', 'd'))
COOCCURRENCE_COUNTS = {('a', 'a'): 2, ('a', 'b'): 2, ('a', 'c'): 1, ('a', 'd'): 1, ('a', 'x'): 1,
                       ('b', 'b'): 2, ('b', 'c'): 1, ('b', 'd'): 1, ('b', 'x'): 1,
                       ('c', 'c'): 2, ('c', 'd'): 2, ('d', 'd'): 2, ('x', 'x'): 1}
COOCCURRENCE_COUNTS_TRACKED_A = COOCCURRENCE_COUNTS.copy()
COOCCURRENCE_COUNTS_TRACKED_A.update({('a+b', 'a+c'): 1, ('a+b', 'a+d'): 1, ('a+b', 'a+x'): 1, ('a+c', 'a+d'): 1})
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

    def test_cooccurrence_counting_explicit_count(self):
        features = (('a', 'b'), ('a', 'b', 'x'))
        p = CooccurrenceProfile.from_feature_lists(features, tracked_features=['x'])
        self.assertEqual(p.attrs['vector_count'], 2)
        self.assertEqual(p.interrelation_value('a', 'b'), 2)
        self.assertEqual(p.interrelation_value('a', 'x'), 1)
        self.assertEqual(p.interrelation_value('b', 'x'), 1)
        self.assertEqual(p.interrelation_value('a+x', 'b+x'), 1)

    def test_cooccurrence_counting_tracked(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES, tracked_features=['a'])
        for feature_tuple, count in COOCCURRENCE_COUNTS_TRACKED_A.items():
            self.assertEqual(p.interrelation_value(feature_tuple[0], feature_tuple[1]), count)
            self.assertEqual(p.interrelation_value(feature_tuple[1], feature_tuple[0]), count)

    def test_features2cooccurrences(self):
        features = ('c', 'a', 'b', 'a')
        reference_cooccurrences = {('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')}
        cooccurrences = set(CooccurrenceProfile.features2cooccurrences(features))
        self.assertSetEqual(cooccurrences, reference_cooccurrences)

    def test_features2cooccurrences_tracked_minimal(self):
        features = ('a', 'b', 'x')
        cooccurrences = set(CooccurrenceProfile.features2cooccurrences(features, tracked_features=['x']))
        self.assertSetEqual(
            cooccurrences, {('a+x', 'b+x'), ('a', 'a'), ('a', 'x'), ('b', 'b'), ('x', 'x'), ('b', 'x'), ('a', 'b')})

    def test_features2cooccurrences_tracked_minimal_sans(self):
        features = ('a', 'b')
        cooccurrences = set(CooccurrenceProfile.features2cooccurrences(features, tracked_features=['x']))
        self.assertSetEqual(
            cooccurrences, {('a', 'a'), ('b', 'b'), ('a', 'b')})

    def test_features2cooccurrences_tracked(self):
        features = ('c', 'a', 'b', 'a', 'x')
        reference_cooccurrences = {('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c'),
                                   ('a', 'x'), ('b', 'x'), ('c', 'x'), ('x', 'x'),
                                   ('a+b', 'a+c'), ('a+b', 'a+x'), ('a+c', 'a+x')}
        cooccurrences = set(CooccurrenceProfile.features2cooccurrences(features, tracked_features=['a']))
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
        p = CooccurrenceProfile.from_dict(p_cooccurrence_counts, vector_count=4) + \
            CooccurrenceProfile.from_dict(q_cooccurrence_counts, vector_count=5)
        r_cooccurrence_counts = {key: value * 2 for key, value in COOCCURRENCE_COUNTS.items()}
        r_cooccurrence_counts.update({('y', 'y'): 1, ('z', 'z'): 2})
        reference_profile = CooccurrenceProfile.from_dict(r_cooccurrence_counts, vector_count=9)
        self.assertEqual(p.attrs['vector_count'], reference_profile.attrs['vector_count'])
        p.df.sort_index(inplace=True)
        reference_profile.df.sort_index(inplace=True)
        self.assertTrue(p.df.equals(reference_profile.df))
        with self.assertRaises(ValueError):
            reference_profile = reference_profile + 'stuff'

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
            features.update(set(feature_tuple))
        self.assertSetEqual(p.distinct_features(), features)
        selection = p.df.loc[p.df.index.get_level_values('feature1') != 'a']
        features.remove('a')
        self.assertSetEqual(p.distinct_features(selection), features)

    def test_feature_interrelations(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        num_max_interrelations = p.num_max_interrelations()
        measured_interrelations = 0
        for f1, f2, value in p.iterate_feature_interrelations():
            measured_interrelations += 1
            self.assertNotEqual(f1, f2)
            self.assertEqual(int(value), COOCCURRENCE_COUNTS.get((f1, f2), 0))
        self.assertEqual(num_max_interrelations, measured_interrelations)

    def test_select_self_relations(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        self_relations = p.select_self_relations()
        num_self_relations = p.num_features()
        measured_self_relations = 0
        for multiindex, value in self_relations.iterrows():
            f1, f2 = multiindex
            measured_self_relations += 1
            self.assertEqual(f1, f2)
            self.assertEqual(int(value.iloc[0]), COOCCURRENCE_COUNTS[(f1, f2)])
        self.assertEqual(num_self_relations, measured_self_relations)

    def test_raw_standard_interrelation_deviation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        interrelation_values = [value for features, value in COOCCURRENCE_COUNTS.items()
                                if features[0] != features[1]]
        interrelation_values_std = statistics.stdev(interrelation_values)
        self.assertEqual(p.standard_raw_interrelation_deviation(), interrelation_values_std)

    def test_standard_self_relation_deviation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        interrelation_values = [value for features, value in COOCCURRENCE_COUNTS.items()
                                if features[0] == features[1]]
        interrelation_values_std = statistics.stdev(interrelation_values)
        self.assertEqual(p.standard_self_relation_deviation(), interrelation_values_std)

    def test_mean_raw_interrelation_value(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        interrelation_values = [value for features, value in COOCCURRENCE_COUNTS.items()
                                if features[0] != features[1]]
        interrelation_values_mean = statistics.mean(interrelation_values)
        self.assertEqual(p.mean_raw_interrelation_value(), interrelation_values_mean)

    def test_mean_self_relation_value(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        interrelation_values = [value for features, value in COOCCURRENCE_COUNTS.items()
                                if features[0] == features[1]]
        interrelation_values_mean = statistics.mean(interrelation_values)
        self.assertEqual(p.mean_self_relation_value(), interrelation_values_mean)

    def test_mean_interrelation_value(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        interrelation_values_sum = sum([value for features, value in COOCCURRENCE_COUNTS.items()
                                        if features[0] != features[1]])
        features = set()
        for feature_tuple in FEATURE_TUPLES:
            features.update(feature_tuple)
        feature_count = len(features)
        interrelation_max_count = (feature_count * feature_count - feature_count) / 2
        mean_interrelation_value = float(interrelation_values_sum) / interrelation_max_count
        self.assertEqual(p.mean_interrelation_value(), mean_interrelation_value)

    def test_standard_interrelation_deviation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        features = set()
        for feature_tuple in FEATURE_TUPLES:
            features.update(feature_tuple)
        feature_count = len(features)
        interrelation_max_count = (feature_count * feature_count - feature_count) / 2
        interrelation_values = [value for features, value in COOCCURRENCE_COUNTS.items() if features[0] != features[1]]
        interrelation_values.extend([0 * i for i in range(int(interrelation_max_count - len(interrelation_values)))])
        interrelation_values_std = statistics.pstdev(interrelation_values)
        self.assertEqual(p.standard_interrelation_deviation(), interrelation_values_std)

    def test_convert_to_zscore(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        prior_imputation_value = p.attrs['imputation_value']
        self_relations_mean = p.mean_self_relation_value()
        self_relations_std = p.standard_self_relation_deviation()
        zscores_self_relations = {features: (value - self_relations_mean) / self_relations_std
                                  for features, value in COOCCURRENCE_COUNTS.items()
                                  if features[0] == features[1]}
        interrelations_mean = p.mean_interrelation_value()
        interrelations_std = p.standard_interrelation_deviation()
        zscores_interrelations = {features: (value - interrelations_mean) / interrelations_std
                                  for features, value in COOCCURRENCE_COUNTS.items()
                                  if features[0] != features[1]}
        p.convert_to_zscore()
        for features, value in zscores_self_relations.items():
            self.assertEqual(p.interrelation_value(features[0], features[1]), value)
        for features, value in zscores_interrelations.items():
            self.assertEqual(p.interrelation_value(features[0], features[1]), value)
        self.assertEqual(p.attrs['imputation_value'],
                         (prior_imputation_value - interrelations_mean) / interrelations_std)

    def test_select_major_self_relations(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        major_self_relations = p.select_major_self_relations(1.0)
        self.assertEqual(len(major_self_relations), 1)
        self.assertEqual(major_self_relations.at[('x', 'x'), 'value'], 1)
        major_self_relations = p.select_major_self_relations(0.0)
        for features, value in p.select_self_relations().iterrows():
            major_value = major_self_relations.at[features, 'value']
            self.assertEqual(major_value, value['value'])

    def test_select_major_interrelations(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        major_interrelations = p.select_major_interrelations(1.0)
        self.assertEqual(len(major_interrelations), 2)
        self.assertEqual(major_interrelations.at[('a', 'b'), 'value'], 2)
        self.assertEqual(major_interrelations.at[('c', 'd'), 'value'], 2)
        major_interrelations = p.select_major_interrelations(0.0)
        for features, value in p.select_raw_interrelations().iterrows():
            major_value = major_interrelations.at[features, 'value']
            self.assertEqual(major_value, value['value'])

    def test_to_explicit_matrix(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        explicit_matrix = p.to_explicit_matrix()
        features = p.distinct_features()
        for feature_a in features:
            for feature_b in features:
                cooccurrence = max((COOCCURRENCE_COUNTS.get((feature_a, feature_b), 0),
                                    COOCCURRENCE_COUNTS.get((feature_b, feature_a), 0)))
                self.assertEqual(explicit_matrix.at[feature_a, feature_b], cooccurrence)

    def test_to_distance_matrix(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)

        def cooccurrence_to_distance(x):
            return 1 / (x + 1)

        distance_matrix = p.to_distance_matrix(distance_conversion_function=cooccurrence_to_distance,
                                               zero_self_relations=False)
        features = p.distinct_features()
        for feature_a in features:
            for feature_b in features:
                cooccurrence = max((COOCCURRENCE_COUNTS.get((feature_a, feature_b), 0),
                                    COOCCURRENCE_COUNTS.get((feature_b, feature_a), 0)))
                distance = cooccurrence_to_distance(cooccurrence)
                self.assertEqual(distance_matrix.at[feature_a, feature_b], distance)
        selection = p.df.loc[p.df.index.get_level_values('feature1') != 'a']
        distance_matrix = p.to_distance_matrix(selection,
                                               distance_conversion_function=cooccurrence_to_distance,
                                               zero_self_relations=True)
        features.remove('a')
        with self.assertRaises(KeyError):
            self.assertEqual(distance_matrix.at['a', 'a'], 0)
        for feature in features:
            self.assertEqual(distance_matrix.at[feature, feature], 0)

    def test_force_string_features(self):
        extended_features = ((1, 2), (2, 3))
        p = CooccurrenceProfile.from_feature_lists(extended_features)
        for f in p.distinct_features():
            self.assertIsInstance(f, str)

    def test_to_csv(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        csv_buffer = StringIO()
        p.to_csv(csv_buffer)
        csv_buffer.seek(0)  # revert to the beginning
        df = pandas.read_csv(csv_buffer)
        q = CooccurrenceProfile.from_dataframe(df)
        self.assertTrue(p.df.equals(q.df))

    def test_merge_features(self):
        feature_lists = [('a', 'b', 'c'), ('a', 'b', 'x'), ('d', 'c'), ('b', 'a', 'a', 'x'), ('b+d', 'a', 'c'),
                         'd+x+a+a']
        expected_features = ['a+b+c', 'a+b+x', 'c+d', 'a+b+x', 'a+b+c+d', 'a+d+x']
        for feature_list, expected_features in zip(feature_lists, expected_features):
            self.assertEqual(CooccurrenceProfile.merge_features(feature_list, delimiter='+'),
                             expected_features)


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

    def test_mean_interrelation_value(self):
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        imputation_probability = 1.0 / (len(FEATURE_TUPLES) + 1)
        interrelation_values = [value / len(FEATURE_TUPLES) for features, value in COOCCURRENCE_COUNTS.items()
                                if features[0] != features[1]]
        interrelation_values_sum = sum(interrelation_values)
        features = set()
        for feature_tuple in FEATURE_TUPLES:
            features.update(feature_tuple)
        feature_count = len(features)
        interrelation_max_count = (feature_count * feature_count - feature_count) / 2
        num_imputed_values = interrelation_max_count - len(interrelation_values)
        interrelation_values_sum += imputation_probability * num_imputed_values
        mean_interrelation_value = float(interrelation_values_sum) / interrelation_max_count
        self.assertEqual(p.mean_interrelation_value(), mean_interrelation_value)

    def test_standard_interrelation_deviation(self):
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        features = set()
        for feature_tuple in FEATURE_TUPLES:
            features.update(feature_tuple)
        feature_count = len(features)
        interrelation_max_count = (feature_count * feature_count - feature_count) / 2
        imputation_probability = 1.0 / (len(FEATURE_TUPLES) + 1)
        interrelation_values = [value for features, value in COOCCURRENCE_PROBABILITIES.items()
                                if features[0] != features[1]]
        interrelation_values.extend([imputation_probability
                                     for i in range(int(interrelation_max_count - len(interrelation_values)))])
        interrelation_values_std = statistics.pstdev(interrelation_values)
        self.assertEqual(p.standard_interrelation_deviation(), interrelation_values_std)

    def test_profile_dataframe_separation(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        q = CooccurrenceProbabilityProfile.from_cooccurrence_profile(p)
        self.assertFalse(p.df is q.df, "The dataframe must not be the same instance, thus affecting previous profile")

    def test_profile_empty_input(self):
        test_tuples_1 = ((), set(), ('a', 'b', 'x'), ('x', 'z_test'))
        test_tuples_2 = ((), set(), ('a', 'b', 'x'), ('[CH](=O)#"{}', 'y_test'))
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)
        q = CooccurrenceProfile.from_feature_lists(test_tuples_1)
        r = CooccurrenceProfile.from_feature_lists(test_tuples_2)
        q.add_another_cooccurrence_profile(p)
        q.add_another_cooccurrence_profile(r)
        z = CooccurrenceProfile.from_feature_lists(
            [feature_set for feature_group in (test_tuples_1, test_tuples_2, FEATURE_TUPLES)
             for feature_set in feature_group])
        z.df.sort_index(inplace=True)
        q.df.sort_index(inplace=True)
        self.assertTrue(q.df.equals(z.df))

    def test_tracked_features_pkld(self):
        p = CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES, tracked_features=['a', 'x'])
        self.assertEqual(p.attrs['tracked_features'], ['a', 'x'])
        q = CooccurrenceProbabilityProfile.from_cooccurrence_profile(p)
        self.assertEqual(q.attrs['tracked_features'], ['a', 'x'])
        test_pkld_contributions = q.tracked_features_pkld(['a', 'b', 'c', 'd', 'x'], get_contributions=True)
        self.assertIsInstance(test_pkld_contributions, dict)
        self.assertEqual(len(test_pkld_contributions), 2)
        self.assertIsInstance(test_pkld_contributions['a'], dict)
        test_pkld_contributions_aggregated = q.tracked_features_pkld(['b', 'c'], get_contributions=False)
        self.assertIsInstance(test_pkld_contributions_aggregated, dict)
        self.assertEqual(len(test_pkld_contributions_aggregated), 2)
        self.assertIsInstance(test_pkld_contributions_aggregated['a'], float)


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

    def test_profile_dataframe_separation(self):
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        q = PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(p)
        self.assertFalse(p.df is q.df, "The dataframe must not be the same instance, thus affecting previous profile")

    def test_pmi_vector_count_requirement(self):
        p = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p.attrs['vector_count'] = None  # destroy the default vector_count data
        with self.assertRaises(ValueError):
            PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(p)
        q = PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(p, vector_count=100)
        self.assertIsInstance(q, PointwiseMutualInformationProfile)

    def test_mean_feature_interrelation_value(self):
        p = PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)))
        tightness = p.mean_feature_interrelation_value(('a', 'b'), omit_self_relations=True)
        self.assertEqual(tightness, p.df.at[('a', 'b'), 'value'])
        tightness = p.mean_feature_interrelation_value(('a', 'b'), omit_self_relations=False)
        self.assertEqual(tightness, p.df.at[('a', 'b'), 'value'] / 3)  # a,a and b,b should be both 0
        tightness = p.mean_feature_interrelation_value(('a', 'b', 'c'))
        self.assertEqual(
            tightness,
            numpy.mean([p.df.at[('a', 'b'), 'value'], p.df.at[('a', 'c'), 'value'], p.df.at[('b', 'c'), 'value']])
        )

    def test_relative_feature_tightness(self):
        p = PointwiseMutualInformationProfile.from_cooccurrence_probability_profile(
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES)))
        tightness = p.relative_feature_tightness(('a', 'b'))
        self.assertEqual(tightness, p.df.at[('a', 'b'), 'value'])
        tightness = p.relative_feature_tightness(('a', 'b', 'c'))
        self.assertEqual(
            tightness,
            numpy.mean([p.df.at[('a', 'b'), 'value'], p.df.at[('a', 'c'), 'value'], p.df.at[('b', 'c'), 'value']])
        )


class TestPointwiseKLDivergenceProfile(unittest.TestCase):
    def test_pointwise_kld_calculation(self):
        cpp1 = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(cpp1, cpp1)
        self.assertTrue((p.df['value'] == 0).all(), "KL divergence with itself should be 0")
        self.assertEqual(p.attrs['imputation_value'], 0, "KL divergence imputation for itself should be 0")
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            cpp1,
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])))
        self.assertEqual(p.interrelation_value('a', 'b'), np.log2(2.0 / 3))
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])),
            cpp1)
        self.assertEqual(cpp1.num_raw_interrelations(), p.num_raw_interrelations())
        self.assertEqual(cpp1.num_max_interrelations(), p.num_max_interrelations())

    def test_relative_feature_divergence(self):
        cpp1 = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            cpp1,
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])))
        divergence = p.relative_feature_divergence(('a', 'b'))
        self.assertEqual(divergence, p.df.at[('a', 'b'), 'value'])


class TestMultilabelPointwiseKLDivergenceProfile(unittest.TestCase):
    pass


class TestPointwiseJeffreysDivergenceProfile(unittest.TestCase):
    def test_pointwise_jd_calculation(self):
        cpp1 = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            CooccurrenceProfile.from_feature_lists(FEATURE_TUPLES))
        p = PointwiseJeffreysDivergenceProfile.from_cooccurrence_probability_profiles(cpp1, cpp1)
        self.assertTrue((p.df['value'] == 0).all(), "KL divergence with itself should be 0")
        self.assertEqual(p.attrs['imputation_value'], 0, "KL divergence imputation for itself should be 0")
        p = PointwiseJeffreysDivergenceProfile.from_cooccurrence_probability_profiles(
            cpp1,
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])))
        self.assertEqual(p.interrelation_value('a', 'b'), abs(np.log2(2.0 / 3.0)) + abs(np.log2(3.0 / 2.0)))
        p = PointwiseJeffreysDivergenceProfile.from_cooccurrence_probability_profiles(
            CooccurrenceProbabilityProfile.from_cooccurrence_profile(
                CooccurrenceProfile.from_feature_lists([('a', 'b')])),
            cpp1)
        self.assertEqual(p.interrelation_value('a', 'b'), abs(np.log2(2.0 / 3.0)) + abs(np.log2(3.0 / 2.0)))
        self.assertEqual(cpp1.num_raw_interrelations(), p.num_raw_interrelations())
        self.assertEqual(cpp1.num_max_interrelations(), p.num_max_interrelations())


if __name__ == '__main__':
    unittest.main()
