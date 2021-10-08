from collections import Counter
from pandas import DataFrame


class InterrelationProfile:
    def __init__(self, df, **kwargs):
        self.df = df
        self.attrs = kwargs


class CooccurrenceProfile(InterrelationProfile):
    @classmethod
    def from_feature_lists(cls, feature_lists):
        cooccurrence_counter = Counter()
        processed_lists = 0
        for feature_list in feature_lists:
            cooccurrence_counter.update(CooccurrenceProfile._features2cooccurrences(feature_list))
            processed_lists += 1
        df = DataFrame.from_dict(cooccurrence_counter, orient='index', columns=['count'])
        return cls(df, vector_count=processed_lists)

    @staticmethod
    def _features2cooccurrences(features, delimiter='|'):
        features = list(set(features))  # get rid of duplicates
        features.sort()
        for i, feature in enumerate(features):
            feature = feature.replace(delimiter, '')
            for other_feature in features[i:]:
                yield f"{feature}{delimiter}{other_feature.replace(delimiter, '')}"


class CooccurrenceProbabilityProfile(InterrelationProfile):
    @classmethod
    def from_cooccurrence_profile(cls, cooccurrence_profile, *args, vector_count=None, **kwargs):
        """Creates a Cooccurrence Probability Profile from the provided Cooccurrence Profile.
        To calculate probability, overall count of feature vector is needed.
        The count is either explicitly provided through the vector_count kwarg, or inferred from
        the cooccurrence_profile.attrs['vector_count'] or, failing that, as a max of its 'count' column"""
        if not vector_count:
            vector_count = cooccurrence_profile.attrs.get('vector_count', cooccurrence_profile.df['count'].max())
        df = cooccurrence_profile.df.divide(vector_count)
        df.rename(columns={'count': 'probability'}, inplace=True)
        kwargs['vector_count'] = vector_count
        return cls(df, **kwargs)
