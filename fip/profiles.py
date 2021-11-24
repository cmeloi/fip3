from collections import Counter

import numpy as np
import pandas
import numpy


class InterrelationProfile:
    def __init__(self, df, *, zscore=False, **kwargs):
        self.df = df
        self.attrs = kwargs
        if zscore:
            self.convert_to_zscore()

    @classmethod
    def from_dict(cls, value_dict, *args, **kwargs):
        df = pandas.DataFrame(((features[0], features[1], value)
                               for features, value in value_dict.items()),
                              columns=['feature1', 'feature2', 'value'])
        df.set_index(['feature1', 'feature2'], inplace=True)
        return cls(df, *args, **kwargs)

    @staticmethod
    def features2cooccurrences(features, *, omit_self_relations=False):
        features = list(set(features))  # get rid of duplicates
        features.sort()
        for i, feature in enumerate(features):
            if omit_self_relations:
                for other_feature in features[i+1:]:
                    yield str(feature), str(other_feature)
            else:
                for other_feature in features[i:]:
                    yield str(feature), str(other_feature)

    def convert_to_zscore(self):
        self.df = (self.df - self.mean_interrelation_value()) / self.standard_interrelation_deviation()

    def interrelation_value(self, f1, f2):
        try:
            if f1 <= f2:
                return self.df.loc[f1, f2]['value']
            else:
                return self.df.loc[f2, f1]['value']
        except KeyError:
            return self._get_imputation_value(f1, f2)

    def features_interrelation_values(self, features):
        # TODO: consider using df.index.intersection for this instead of loop
        for feature, other_feature in self.features2cooccurrences(features):
            yield self.interrelation_value(feature, other_feature)

    def distinct_features(self):
        return set(self.df.index.unique(level='feature1'))

    def feature_self_relations(self):
        return {feature: self.interrelation_value(feature, feature) for feature in self.distinct_features()}

    def feature_interrelations(self, *args, omit_self_relations=False):
        raise NotImplementedError

    def mean_interrelation_value(self):
        #  interrelation_sum += self._get_imputation_value(f1, f2)  # TODO: multiply by implicit interrelation count
        #  return interrelation_sum / self.num_interrelations()
        raise NotImplementedError

    def mean_raw_interrelation_value(self):
        return float(sum(self.df['value']))

    def num_interrelations(self):
        # all theoretically possible interrelation for the current feature set
        num_distinct_features = len(self.distinct_features())
        return (num_distinct_features * num_distinct_features - num_distinct_features) / 2

    def num_raw_interrelations(self):
        return len(self.df.index) - len(self.distinct_features())

    def standard_interrelation_deviation(self):
        raise NotImplementedError

    def standard_raw_interrelation_deviation(self):
        explicit_values = self.df['value']
        return np.std(explicit_values)

    def _get_imputation_value(self, f1, f2):
        raise NotImplementedError  # to be overridden by child classes

    def __getitem__(self, feature_list):
        return self.features_interrelation_values(feature_list)


class CooccurrenceProfile(InterrelationProfile):
    @classmethod
    def from_feature_lists(cls, feature_lists, *args, **kwargs):
        cooccurrence_counter = Counter()
        processed_lists = 0
        for feature_list in feature_lists:
            cooccurrence_counter.update(CooccurrenceProfile.features2cooccurrences(feature_list))
            processed_lists += 1
        df = pandas.DataFrame(((features[0], features[1], value)
                               for features, value in cooccurrence_counter.items()),
                              columns=['feature1', 'feature2', 'value'])
        df.set_index(['feature1', 'feature2'], inplace=True)
        kwargs['vector_count'] = processed_lists
        kwargs['imputation_value'] = 0
        return cls(df, *args, **kwargs)

    @classmethod
    def from_feature_lists_split_on_feature(cls, iterable, feature):
        """Generates and returns two CooccurrenceProbabilityProfile instances from the provided iterable,
        and the provided feature. Returns main profile from feature vectors containing the given feature,
        and a reference profile from all other feature vectors"""
        positive_counter, positive_vectors = Counter(), 0
        negative_counter, negative_vectors = Counter(), 0
        for feature_list in iterable:
            if feature in feature_list:
                positive_counter.update(cls.features2cooccurrences(feature_list))
                positive_vectors += 1
            else:
                negative_counter.update(cls.features2cooccurrences(feature_list))
                negative_vectors += 1
        return (cls.from_dict(positive_counter, vector_count=positive_vectors),
                cls.from_dict(negative_counter, vector_count=negative_vectors))

    def _get_imputation_value(self, f1, f2):
        return 0

    def __add__(self, other):
        self.df = self.df.add(other.df, fill_value=0)
        self.df['value'] = self.df['value'].astype(int)
        self.attrs['vector_count'] += other.attrs['vector_count']
        return self


class CooccurrenceProbabilityProfile(InterrelationProfile):
    @classmethod
    def from_cooccurrence_profile(cls, cooccurrence_profile, *args, vector_count=None, **kwargs):
        """Creates a Cooccurrence Probability Profile from the provided Cooccurrence Profile.
        To calculate probability, overall count of feature vector is needed.
        The count is either explicitly provided through the vector_count kwarg, or inferred from
        the cooccurrence_profile.attrs['vector_count'] or, failing that, as a max of its 'count' column

        :param cooccurrence_profile: the source CooccurrenceProfile instance
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param vector_count: explicit count of feature vectors, i.e. samples, to manually adjust the probabilities
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: a CooccurrenceProbabilityProfile instance
        """
        if not vector_count:
            vector_count = cooccurrence_profile.attrs.get('vector_count', cooccurrence_profile.df['value'].max())
        df = cooccurrence_profile.df.divide(vector_count)
        kwargs['vector_count'] = vector_count
        kwargs['imputation_value'] = 1.0 / (vector_count + 1)  # "most-optimist" imputation value
        return cls(df, *args, **kwargs)

    def _get_imputation_value(self, f1, f2):
        return self.attrs['imputation_value']


class PointwiseMutualInformationProfile(InterrelationProfile):
    @classmethod
    def from_cooccurrence_probability_profile(cls, cooccurrence_probability_profile,
                                              *args, vector_count=None, **kwargs):
        """Generate a PMI interrelation profile.

        :param cooccurrence_probability_profile: the source CooccurrenceProbabilityProfile instance
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param vector_count: explicit count of feature vectors, i.e. samples, to manually adjust the probabilities
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: PointwiseMutualInformationProfile instance
        """
        if not vector_count:
            vector_count = cooccurrence_probability_profile.attrs.get('vector_count')
        kwargs['vector_count'] = vector_count
        standalone_probabilities = cooccurrence_probability_profile.feature_self_relations()
        kwargs['imputation_probability'] = 1.0 / (vector_count + 1)  # "most-optimist" imputation value
        kwargs['imputation_marginal_probabilities'] = {feature: (feature_probability*vector_count + 1) / (vector_count + 1)
                                                       for feature, feature_probability
                                                       in standalone_probabilities.items()}
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x / (standalone_probabilities[x.name[0]] * standalone_probabilities[x.name[1]]))
            if x.name[0] != x.name[1] else x * 0,
            axis=1)  # the if/else clause because P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        return cls(df, *args, **kwargs)

    def _get_imputation_value(self, feature1, feature2):
        """PMI imputation based on "most-optimistic" scenario that the co-occurrence would happen in the n+1 sample

        :param feature1: First feature for PMI imputation
        :param feature2: Second feature for PMI imputation
        :return: The pair PMI based on imputed values
        """
        if feature1 == feature2:
            return 0  # P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        generic_imputation_probability = self.attrs['imputation_probability']
        feature1_imputation_probability = self.attrs['imputation_marginal_probabilities'].get(feature1, generic_imputation_probability)
        feature2_imputation_probability = self.attrs['imputation_marginal_probabilities'].get(feature2, generic_imputation_probability)
        return generic_imputation_probability / (feature1_imputation_probability * feature2_imputation_probability)


class PointwiseKLDivergenceProfile(InterrelationProfile):
    @classmethod
    def from_cooccurrence_probability_profiles(cls, cooccurrence_probability_profile, reference_probability_profile,
                                               *args, vector_count=None, **kwargs):
        kwargs['vector_count'] = vector_count
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x / reference_probability_profile.interrelation_value(x.name[0], x.name[1])),
            axis=1)
        df.dropna(inplace=True)
        return cls(df, *args, **kwargs)
