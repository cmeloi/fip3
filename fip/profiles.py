from collections import Counter
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas
import numpy


class InterrelationProfile(object):
    """A generic parent class representing an interrelation profile, and implementing their common functionality.
    Not meant for instantiation."""
    __metaclass__ = ABCMeta

    def __init__(self, df, *, zscore=False, **kwargs):
        self.df = df
        self.attrs = kwargs
        if zscore:
            self.convert_to_zscore()

    @classmethod
    def from_dict(cls, value_dict, *args, **kwargs):
        """Loads an interrelation profile from a dictionary of {(feature1, feature2): value}

        :param value_dict: the dictionary to load
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: the corresponding InterrelationProfile instance
        """
        df = pandas.DataFrame(((features[0], features[1], value)
                               for features, value in value_dict.items()),
                              columns=['feature1', 'feature2', 'value'])
        df.set_index(['feature1', 'feature2'], inplace=True)
        return cls(df, *args, **kwargs)

    @staticmethod
    def features2cooccurrences(features):
        """Processes an iterable of features into a set of feature co-occurrences.

        :param features: an iterable of features to process, e.g. (feature1, feature2, feature4, ...)
        :return: a co-occurrence generator
        """
        features = list(set(features))  # get rid of duplicates
        features.sort()
        for i, feature in enumerate(features):
            for other_feature in features[i:]:
                yield str(feature), str(other_feature)

    def select_self_relations(self):
        """Provides all feature self-relations within the profile as a DataFrame subset selection.

        :return: feature self-relations as a Pandas DataFrame subset
        """
        return self.df.loc[self.df.index.get_level_values('feature1') == self.df.index.get_level_values('feature2')]

    def select_raw_interrelations(self):
        """Provides all explicit (i.e. not imputed) feature interrelations (i.e. not self-relations) within the profile
        as a DataFrame subset selection.

        :return: feature interrelations as a Pandas DataFrame subset
        """
        return self.df.loc[self.df.index.get_level_values('feature1') != self.df.index.get_level_values('feature2')]

    def select_all(self):
        """Provides all explicit feature relations (both self-relations and interrelations) within the profile
        as a DataFrame. The DataFrame itself is also directly accessible through InterrelationProfile.df

        :return: all feature relations as a Pandas DataFrame
        """
        return self.df

    def self_relations_dict(self):
        """Returns self-relation values of all features in the profile as a dictionary.

        :return: a dictionary of {feature: self_interrelation_value} pairs
        """
        return {str(multiindex[0]): float(value) for multiindex, value in self.select_self_relations().iterrows()}

    def convert_to_zscore(self):
        """Converts the values within the InterrelationProfile into Z-scores, i.e. subtracts mean,
        divides by standard deviation.

        :return: None, the InterrelationProfile is changed itself
        """
        # TODO: Make split Z-score for self-relations and interrelations
        self.df = (self.df - self.mean_interrelation_value()) / self.standard_interrelation_deviation()

    def interrelation_value(self, f1, f2=None):
        """Returns the interrelation value for the feature pair provided in the arguments.
        If second argument f2 is not filled in, returns self-relation (f1, f1).

        :param f1: the first feature
        :param f2: the second feature, default None
        :return: The interrelation value between the two features within the profile. Usually int or float.
        """
        if f2 is None:
            return self.df.at[(f1, f1), 'value']
        try:
            if f1 <= f2:
                return self.df.at[(f1, f2), 'value']
            else:
                return self.df.at[(f2, f1), 'value']
        except KeyError:
            return self._get_imputation_value(f1, f2)

    def features_interrelation_values(self, features):
        """Provides interrelation values within the profile for all features within a given feature list.

        :param features: features to look up within the profile
        :return: a generator yielding the interrelation values, usually floats or ints
        """
        # TODO: consider using df.index.intersection for this instead of loop
        for feature, other_feature in self.features2cooccurrences(features):
            yield self.interrelation_value(feature, other_feature)

    def distinct_features(self):
        """Provides a set of all distinct features present within the interrelation profile.

        :return: feature names as a set of strings
        """
        return set(self.df.index.unique(level='feature1'))

    def feature_interrelations(self):
        """Yields all explicit interrelations values between features in the profile, in a tuple.
        Omits self-relations of features.

        :return: yields tuples of (feature1, feature2, interrelation_value)
        """
        for multiindex, value in self.df.iterrows():
            if multiindex[0] == multiindex[1]:
                continue
            yield multiindex[0], multiindex[1], value

    @abstractmethod
    def mean_interrelation_value(self):
        """Provides mean value of all interrelation values within the profile, including imputed values.
        Implemented individually within different FeatureInterrelation types, due to imputation differences.
        """
        #  interrelation_sum += self._get_imputation_value(f1, f2)  # TODO: multiply by implicit interrelation count
        #  return interrelation_sum / self.num_interrelations()
        raise NotImplementedError

    def mean_raw_interrelation_value(self):
        """Provides mean value of all explicit interrelation values within the profile, i.e. not counting in
        the imputation values.

        :return: the mean explicit interrelation value as a float
        """
        num_raw_interrelations = 0
        running_sum = 0.0
        for f1, f2, value in self.feature_interrelations():
            num_raw_interrelations += 1
            running_sum += value
        return running_sum / num_raw_interrelations

    def num_max_interrelations(self):
        """Provides the count of all possible feature interrelations that can exist within the profile,
        based solely on the amount of observed features.

        :return: count of all possible interrelations
        """
        num_distinct_features = len(self.distinct_features())
        return (num_distinct_features * num_distinct_features - num_distinct_features) / 2

    def num_raw_interrelations(self):
        """Provides the count of all interrelations that explicitly occur within the profile, i.e. all interrelations
        that are non-imputed, non-self-relations.

        :return: count of all explicit interrelation values
        """
        return len(self.df.index) - len(self.distinct_features())

    def num_features(self):
        """Provides the count of all features (individual features, not their interrelations)
        that occur within the profile.

        :return: count of all features
        """
        return len(self.distinct_features())

    @abstractmethod
    def standard_interrelation_deviation(self):
        """Provides standard deviation for all interrelation values within the profile, including imputed values.
        Implemented individually within different FeatureInterrelation types, due to imputation differences.
        """
        raise NotImplementedError

    def standard_self_relation_deviation(self):
        """Provides standard deviation from all self-relation values within the profile.
        Ignores interrelations.

        :return: The standard deviation as a float
        """
        explicit_values = self.select_self_relations()['value']
        return float(explicit_values.std())

    def raw_standard_interrelation_deviation(self):
        """Provides standard deviation from all explicit (i.e. non-imputed) interrelation values within the profile.
        Ignores self-relations.

        :return: The standard deviation as a float
        """
        explicit_values = self.select_raw_interrelations()['value']
        return float(explicit_values.std())

    def to_csv(self, target_file):
        """Export the interrelation matrix to a CSV file

        :param target_file: the path to the export
        :return: None
        """
        return self.df.to_csv(target_file)

    @abstractmethod
    def _get_imputation_value(self, f1, f2):
        raise NotImplementedError

    def __getitem__(self, f1, f2=None):
        return self.interrelation_value(f1, f2)


class CooccurrenceProfile(InterrelationProfile):
    """A feature Co-occurrence profile, which is usually a core profile for further interrelation analysis.
    Holds raw counts on how many times did the features occur and co-occur within the characterized set.
    """
    @classmethod
    def from_feature_lists(cls, feature_lists, *args, **kwargs):
        """Generate a Co-occurrence profile from an iterable of feature lists.

        :param feature_lists: the iterable of feature lists to derive the CooccurrenceProfile from
        :param args: any further arguments to be passed to the CooccurrenceProfile init
        :param kwargs: any further keyword arguments to be passed to the CooccurrenceProfile init
        :return: the corresponding CooccurrenceProfile instance
        """
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
        and a reference profile from all other feature vectors.

        :param iterable: the iterable of feature lists to derive the CooccurrenceProfile from
        :param feature: the feature to split the set on
        :return: a tuple of the CooccurrenceProfile instances (with_feature, without_feature)
        """
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

    def _get_imputation_value(self, *args):
        return 0

    def __add__(self, other):
        self.df = self.df.add(other.df, fill_value=0)
        self.df['value'] = self.df['value'].astype(int)
        self.attrs['vector_count'] += other.attrs['vector_count']
        return self


class CooccurrenceProbabilityProfile(InterrelationProfile):
    """A co-occurrence probability profile, holding information about the observed probabilities for feature pair
    co-occurrences within the characterized set."""
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
        kwargs['imputation_probability'] = 1.0 / (vector_count + 1)  # "most-optimist" imputation value
        return cls(df, *args, **kwargs)

    def imputable_standalone_probabilities(self):
        """Calculates standalone probabilities for individual features, to be used in the "most-optimistic" imputation
        scheme, i.e. presuming that the n+1 feature vector will contain all observed features co-occurring.

        :return: a dictionary of feature:probability
        """
        standalone_probabilities = self.self_relations_dict()
        vector_count = self.attrs['vector_count']
        feature2imputable_standalone_probability = {feature: (feature_probability*vector_count + 1) / (vector_count + 1)
                                                    for feature, feature_probability in standalone_probabilities.items()}
        return feature2imputable_standalone_probability

    def _get_imputation_value(self, *args):
        return self.attrs['imputation_probability']


class PointwiseMutualInformationProfile(InterrelationProfile):
    """An interrelation profile consisting of Pointwise Mutual Information (PMI) values observed between the features
    present in the characterized set. PMI is a measure of association between two features - a ratio between the
    observed co-occurrence probability of the feature pair, versus the projected co-occurrence if the features
    were completely independent, based on their individual occurrence rate."""
    @classmethod
    def from_cooccurrence_probability_profile(cls, cooccurrence_probability_profile, *args, **kwargs):
        """Generate a PMI interrelation profile.

        :param cooccurrence_probability_profile: the source CooccurrenceProbabilityProfile instance
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: PointwiseMutualInformationProfile instance
        """
        kwargs['vector_count'] = cooccurrence_probability_profile.attrs['vector_count']
        kwargs['imputation_probability'] = cooccurrence_probability_profile.attrs['imputation_probability']
        kwargs['imputation_standalone_probabilities'] = cooccurrence_probability_profile.imputable_standalone_probabilities()
        standalone_probabilities = cooccurrence_probability_profile.self_relations_dict()
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
        feature1_imputation_probability = self.attrs['imputation_standalone_probabilities'].get(feature1, generic_imputation_probability)
        feature2_imputation_probability = self.attrs['imputation_standalone_probabilities'].get(feature2, generic_imputation_probability)
        return generic_imputation_probability / (feature1_imputation_probability * feature2_imputation_probability)


class PointwiseKLDivergenceProfile(InterrelationProfile):
    """An interrelation profile consisting of pointwise Kullback–Leibler divergence values, a measure of statistical
    distances for each feature pair, between its observed co-occurrence in the characterized set and its observed
    co-occurrence in a reference set.
    """
    @classmethod
    def from_cooccurrence_probability_profiles(cls, cooccurrence_probability_profile, reference_probability_profile, *args, **kwargs):
        """Creates a pointwise KL Divergence interrelation profile quantifying how well do the co-occurrence
        probabilities in the given interrelation profile match those in the given reference interrelation profile.

        pKLD(F1|F2) = P(F1|F2) / Q(F1|F2)

        where F1 and F2 are observed features, P(F1|F2) is their co-occurrence probability within the evaluated
        interrelation profile, and Q(F1|F2) is the same within the reference interrelation profile.

        :param cooccurrence_probability_profile: the CooccurrenceProbabilityProfile instance to be evaluated
        :param reference_probability_profile: the CooccurrenceProbabilityProfile instance to serve as a reference
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: PointwiseKLDivergenceProfile instance
        """
        kwargs['imputation_probability_main'] = cooccurrence_probability_profile.attrs['imputation_probability']
        kwargs['imputation_probability_ref'] = reference_probability_profile.attrs['imputation_probability']
        kwargs['imputation_value'] = kwargs['imputation_probability_main'] / kwargs['imputation_probability_ref']
        # TODO: do merging of the profiles, with partial imputation for pairs that occur in at least one
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x / reference_probability_profile.interrelation_value(x.name[0], x.name[1])), axis=1)
        df.dropna(inplace=True)
        return cls(df, *args, **kwargs)

    def _get_imputation_value(self, *args):
        """PKLD imputation for the case that the features do not co-occur in neither the evaluated, nor the reference
         interrelation profile. It is based on the imputation probability for the individual feature profiles.

        :return: Imputation PKLD for feature co-occurrence appearing in neither profile
        """
        return self.attrs['imputation_value']
