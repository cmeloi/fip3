from collections import Counter

import pandas
import numpy


def generate_profile_pair_for_feature(iterable, feature):
    """Generates and returns two CooccurrenceProbabilityProfile instances from the provided iterable,
    and the provided feature. Returns main profile from feature vectors containing the given feature,
    and a reference profile from all other feature vectors"""
    raise NotImplementedError


class InterrelationProfile:
    def __init__(self, df, *args, **kwargs):
        self.df = df
        self.args = args
        self.attrs = kwargs
        self.imputations = {'best-case': 0}  # TODO: implement best-case imputation

    def get_feature_relation(self, f1, f2, imputation='best-case'):
        try:
            if f1 <= f2:
                return self.df.loc[f1, f2]['value']
            else:
                return self.df.loc[f2, f1]['value']
        except KeyError:
            return self.imputations.get(imputation, pandas.NA)


class CooccurrenceProfile(InterrelationProfile):
    @classmethod
    def from_feature_lists(cls, feature_lists):
        cooccurrence_counter = Counter()
        processed_lists = 0
        for feature_list in feature_lists:
            cooccurrence_counter.update(CooccurrenceProfile._features2cooccurrences(feature_list))
            processed_lists += 1
        df = pandas.DataFrame(((features[0], features[1], value)
                               for features, value in cooccurrence_counter.items()),
                              columns=['feature1', 'feature2', 'value'])
        df.set_index(['feature1', 'feature2'], inplace=True)
        return cls(df, vector_count=processed_lists)

    @staticmethod
    def _features2cooccurrences(features):
        features = list(set(features))  # get rid of duplicates
        features.sort()
        for i, feature in enumerate(features):
            for other_feature in features[i:]:
                yield str(feature), str(other_feature)


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
        standalone_probabilities = {feature: cooccurrence_probability_profile.df.loc[feature, feature]['value']
                                    for feature, feature2 in cooccurrence_probability_profile.df.index.values}
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x/(standalone_probabilities[x.name[0]]*standalone_probabilities[x.name[1]]))
            if x.name[0] != x.name[1] else x*0,
            axis=1)  # the if/else clause because P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        return cls(df, *args, **kwargs)
