from collections import Counter

import numpy as np
import pandas
import numpy


class InterrelationProfile:
    def __init__(self, df, *, imputation=False, zscore=False, **kwargs):
        self.df = df
        self.attrs = kwargs
        self.imputation = imputation
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
            if self.imputation:
                return self._get_imputation_value(f1, f2)
            else:
                return np.NaN

    def features_interrelation_values(self, features):
        for feature, other_feature in self.features2cooccurrences(features):
            yield self.interrelation_value(feature, other_feature)

    def distinct_features(self):
        return set(self.df.index.unique(level='feature1'))

    def feature_self_relations(self):
        return {feature: self.interrelation_value(feature, feature) for feature in self.distinct_features()}

    def feature_interrelations(self, *args, omit_self_relations=False):
        raise NotImplementedError

    def mean_interrelation_value(self):
        interrelation_sum = float(sum(self.df['value']))
        if self.imputation:
            # TODO: finish imputed variant
            raise NotImplementedError
            interrelation_sum += self._get_imputation_value(f1, f2)  # TODO: multiply by implicit interrelation count
        return interrelation_sum / self.num_interrelations()

    def num_interrelations(self):
        num_distinct_features = len(self.distinct_features())
        if self.imputation:  # all theoretically possible interrelation for the current feature set
            return (num_distinct_features * num_distinct_features - num_distinct_features) / 2
        else:  # actual observed interrelations, sans self-interrelations
            return len(self.df.index) - num_distinct_features

    def standard_interrelation_deviation(self, imputation=False):
        explicit_values = self.df['value']
        if imputation:  # also include imputation values for whichever interrelations are not present
            raise NotImplementedError
        else:  # use present values only
            return np.std(explicit_values)

    def _get_imputation_value(self, feature1, feature2):
        raise NotImplementedError  # to be overridden by a specific profile class

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

    def _get_imputation_value(self, feature1, feature2):
        return 0  # the co-occurrences not listed in the values have never happened in the set

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
        the cooccurrence_profile.attrs['vector_count'] or, failing that, as a max of its 'count' column"""
        if not vector_count:
            vector_count = cooccurrence_profile.attrs.get('vector_count', cooccurrence_profile.df['value'].max())
        df = cooccurrence_profile.df.divide(vector_count)
        kwargs['vector_count'] = vector_count
        return cls(df, *args, **kwargs)


class PointwiseMutualInformationProfile(InterrelationProfile):
    @classmethod
    def from_cooccurrence_probability_profile(cls, cooccurrence_probability_profile,
                                              *args, vector_count=None, **kwargs):
        kwargs['vector_count'] = vector_count
        standalone_probabilities = cooccurrence_probability_profile.feature_self_relations()
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x / (standalone_probabilities[x.name[0]] * standalone_probabilities[x.name[1]]))
            if x.name[0] != x.name[1] else x * 0,
            axis=1)  # the if/else clause because P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        return cls(df, *args, **kwargs)

    def _get_imputation_value(self, feature1, feature2):
        # TODO: Implement imputation based on "most-optimistic" scenario
        # that the co-occurrence would happen in the n+1 sample
        raise NotImplementedError


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
