from collections import Counter
from pandas import DataFrame


class InterrelationProfile:
    def __init__(self, df):
        self.df = df


class CooccurrenceProfile(InterrelationProfile):
    @classmethod
    def from_feature_lists(cls, feature_lists):
        cooccurrence_counter = Counter()
        processed_lists = 0
        for feature_list in feature_lists:
            cooccurrence_counter.update(CooccurrenceProfile._features2cooccurrences(feature_list))
            processed_lists += 1
        df = DataFrame.from_dict(cooccurrence_counter, orient='index', columns=['count'])
        df.attrs['vector_count'] = processed_lists
        return cls(df)

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
    def from_cooccurrence_profile(cls, cooccurrence_profile):
        df = cooccurrence_profile.df.divide(cooccurrence_profile.df.attrs['vector_count'])
        df.rename(columns={'count': 'probability'}, inplace=True)
        return cls(df)
