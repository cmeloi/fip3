from abc import ABCMeta, abstractmethod
from collections import Counter
from functools import partial

import numpy
import pandas


class InterrelationProfile(object):
    """A generic parent class representing an interrelation profile, and implementing their common functionality.
    Not meant for instantiation."""
    __metaclass__ = ABCMeta

    def __init__(self, df, *, zscore=False, min_cutoff_value=None, **kwargs):
        if min_cutoff_value:
            df = df[df.value >= min_cutoff_value]
        self.df = df
        self.attrs = kwargs
        if zscore:
            self.convert_to_zscore()

    @classmethod
    def from_dict(cls, value_dict, *args, **kwargs):
        """Loads an interrelation profile from a dictionary of {(feature1, feature2): value}.
        feature1, feature2 are handled as strings and serve as a multiindex for the value.

        :param value_dict: the dictionary to load
        :param zscore: whether to produce a z-scored profile instead of raw values. Keyword argument, default False.
        :param min_cutoff_value: if defined, drop all interrelations with values below the given limit. Default None.
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: the corresponding InterrelationProfile instance
        """
        df = pandas.DataFrame(((str(features[0]), str(features[1]), value)
                               for features, value in value_dict.items()),
                              columns=['feature1', 'feature2', 'value'])
        df.set_index(['feature1', 'feature2'], inplace=True)
        return cls(df, *args, **kwargs)

    @classmethod
    def from_dataframe(cls, dataframe, *args, **kwargs):
        """Loads an interrelation profile from a dataframe containing 'feature1', 'feature2' and 'value' columns.

        :param dataframe: the dataframe to load
        :param zscore: whether to produce a z-scored profile instead of raw values. Keyword argument, default False.
        :param min_cutoff_value: if defined, drop all interrelations with values below the given limit. Default None.
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: the corresponding InterrelationProfile instance
        """
        dataframe.feature1.apply(str)
        dataframe.feature2.apply(str)
        dataframe.set_index(['feature1', 'feature2'], inplace=True)
        return cls(dataframe, *args, **kwargs)

    @staticmethod
    def features2cooccurrences(features, *, omit_self_relations=False):
        """Processes an iterable of features into a string set of feature co-occurrences.

        :param features: an iterable of features to process, e.g. (feature1, feature2, feature4, ...)
        :param omit_self_relations: Whether to ignore feature self-relations, i.e. (f1, f1). Default False.
        :return: a co-occurrence generator
        """
        features = list(set((str(f) for f in features)))  # force strings, get rid of duplicates
        features.sort()
        offset = 0
        if omit_self_relations:
            offset = 1
        for i, feature in enumerate(features):
            for other_feature in features[i + offset:]:
                yield str(feature), str(other_feature)

    @staticmethod
    def row_zscore(mean, standard_deviation, row, *, input_column_name='value', output_column_name='value'):
        """A drop-in static method to calculate Z-score from a value, mean and standard deviation.
        Z-score, aka. standard score, is the number of standard deviations a given value is from the mean.

        :param mean: the mean value within the distribution
        :param standard_deviation: the standard deviation within the distribution
        :param row: a Pandas row to calculate the z score for
        :param input_column_name: column name for the processed value, default 'value'
        :param output_column_name: column name for the output z-score value, default 'value' (i.e. overwrite)
        :return: modified row containing the z-score
        """
        row[output_column_name] = (row[input_column_name] - mean) / standard_deviation
        return row

    def select_self_relations(self, selection=None):
        """Provides all feature self-relations within the profile as a DataFrame subset selection.

        :type selection: the subset to select the self-relations from. Default None, i.e. using the whole dataframe.
        :return: feature self-relations as a Pandas DataFrame subset
        """
        if selection is None:
            selection = self.df
        return selection.loc[selection.index.get_level_values('feature1')
                             == selection.index.get_level_values('feature2')]

    def select_raw_interrelations(self, selection=None):
        """Provides all explicit (i.e. not imputed) feature interrelations (i.e. not self-relations) within the profile
        as a DataFrame subset selection.

        :type selection: the subset to select raw interrelations from. Default None, i.e. using the whole dataframe.
        :return: feature interrelations as a Pandas DataFrame subset
        """
        if selection is None:
            selection = self.df
        return selection.loc[selection.index.get_level_values('feature1')
                             != selection.index.get_level_values('feature2')]

    def select_raw_interrelations_involving(self, features, depth=0):
        """Selects all raw interrelations associated with the given feature or group of features, either directly
        (depth = 0) or by extension through other relations (depth of 1, 2, ...)

        :param features: feature or a list of features to gain features from
        :param depth: how far the interrelations are tracked. Default 0, i.e. only direct interrelations.
        :return: the associated interrelations as a Pandas DataFrame subset
        """
        if isinstance(features, str):
            features = [features]
        features = set(features)
        seek_depth = depth
        while seek_depth > 0:
            seek_depth = seek_depth - 1
            selection = pandas.concat((self.df[self.df.index.get_level_values('feature1').isin(features)],
                                       self.df[self.df.index.get_level_values('feature2').isin(features)]))
            features.update(self.distinct_features(selection))
        selection = pandas.concat((self.df[self.df.index.get_level_values('feature1').isin(features)],
                                   self.df[self.df.index.get_level_values('feature2').isin(features)]))
        return self.select_raw_interrelations(selection[~selection.index.duplicated(keep='first')])

    def select_all(self):
        """Provides all explicit feature relations (both self-relations and interrelations) within the profile
        as a DataFrame. The DataFrame itself is also directly accessible through InterrelationProfile.df

        :return: all feature relations as a Pandas DataFrame
        """
        return self.df

    def select_major_self_relations(self, zscore_cutoff=1.0):
        """Provides all explicit feature self relations within the profile, that are higher or lower than the profile
        average by amount of standard deviations provided by the zscore_cutoff value.
        In other words, the zcore_cutoff is the relative relation strength cutoff based on the amount of standard
        deviations for the individual interrelation values from their mean.

        Returns a selection of the DataFrame. The DataFrame itself is also directly accessible through
        InterrelationProfile.df

        :param zscore_cutoff: Relative relation strength cutoff value. Default 1.0
        :return: major feature self-relations as a Pandas DataFrame
        """
        mean = self.mean_self_relation_value()
        standard_deviation = self.standard_self_relation_deviation()
        lower_cutoff = mean - zscore_cutoff * standard_deviation
        higher_cutoff = mean + zscore_cutoff * standard_deviation
        self_relations = self.select_self_relations()
        return self_relations.loc[
            (self_relations['value'] <= lower_cutoff) | (self_relations['value'] >= higher_cutoff)]

    def select_major_interrelations(self, zscore_cutoff=1.0):
        """Provides all explicit feature interrelations within the profile, that are higher or lower than the profile
        average by amount of standard deviations provided by the zscore_cutoff value.
        In other words, the zcore_cutoff is the relative relation strength cutoff based on the amount of standard
        deviations for the individual interrelation values from their mean.

        Returns a selection of the DataFrame. The DataFrame itself is also directly accessible through
        InterrelationProfile.df

        :param zscore_cutoff: Relative relation strength cutoff value. Default 1.0
        :return: major feature interrelations as a Pandas DataFrame
        """
        mean = self.mean_raw_interrelation_value()
        standard_deviation = self.standard_raw_interrelation_deviation()
        lower_cutoff = mean - zscore_cutoff * standard_deviation
        higher_cutoff = mean + zscore_cutoff * standard_deviation
        interrelations = self.select_raw_interrelations()
        return interrelations.loc[
            (interrelations['value'] <= lower_cutoff) | (interrelations['value'] >= higher_cutoff)]

    def self_relations_dict(self):
        """Returns self-relation values of all features in the profile as a dictionary.

        :return: a dictionary of {feature: self_interrelation_value} pairs
        """
        return {str(multiindex[0]): float(columns['value'])
                for multiindex, columns in self.select_self_relations().iterrows()}

    def convert_to_zscore(self):
        """Converts the values within the InterrelationProfile into Z-scores, i.e. subtracts mean,
        divides by standard deviation.

        :return: None, the InterrelationProfile is changed itself
        """
        self_relations = self.select_self_relations().astype(float)
        self_relations_mean = self.mean_self_relation_value()
        self_relations_standard_deviation = self.standard_self_relation_deviation()
        row_zscore_partial = partial(self.row_zscore, self_relations_mean, self_relations_standard_deviation)
        self_relations_z_scores = self_relations.apply(row_zscore_partial, axis=1)
        self.df.update(self_relations_z_scores, overwrite=True)
        interrelations = self.select_raw_interrelations()
        interrelations_mean = self.mean_interrelation_value()
        interrelations_standard_deviation = self.standard_interrelation_deviation()
        row_zscore_partial = partial(self.row_zscore, interrelations_mean, interrelations_standard_deviation)
        interrelations_z_scores = interrelations.apply(row_zscore_partial, axis=1)
        self.df.update(interrelations_z_scores, overwrite=True)
        self.attrs['imputation_value'] = ((float(self.get_imputation_value(None, None)) - interrelations_mean)
                                          / interrelations_standard_deviation)

    def interrelation_value(self, f1, f2=None):
        """Returns the interrelation value for the feature pair provided in the arguments.
        If second argument f2 is not filled in, returns self-relation (f1, f1).

        :param f1: the first feature
        :param f2: the second feature, default None
        :return: The interrelation value between the two features within the profile. Usually int or float.
        """
        if f2 is None:
            f2 = f1
        try:
            if f1 <= f2:
                return self.df.at[(f1, f2), 'value']
            else:
                return self.df.at[(f2, f1), 'value']
        except KeyError:
            return self.get_imputation_value(f1, f2)

    def features_interrelation_values(self, features, *, omit_self_relations=False):
        """Yields interrelation values within the profile for all features within a given feature list.
        Includes imputed values.

        :param features: features to look up within the profile
        :param omit_self_relations: whether to omit self-relations in the lookup, default False.
        :return: a generator yielding the interrelation values, usually floats or ints
        """
        # TODO: consider using df.index.intersection for this instead of loop
        for feature, other_feature in self.features2cooccurrences(features, omit_self_relations=omit_self_relations):
            yield self.interrelation_value(feature, other_feature)

    def mean_feature_interrelation_value(self, features, *, omit_self_relations=True):
        """Returns the mean interrelation value within the profile for all features within a given feature list.
        Corresponds to feature "tightness" measure for profiles such as (Z)PMI and (Z)PKLD.
        Includes imputed values.

        :param features: features to look up within the profile
        :param omit_self_relations: whether to omit self-relations in the lookup, default True.
        :return: mean interrelation value, as a float
        """
        hit_interrelation_values = self.features_interrelation_values(features, omit_self_relations=omit_self_relations)
        return numpy.mean(list(hit_interrelation_values))

    def distinct_features(self, selection=None):
        """Provides a set of all distinct features present within the interrelation profile.
        Optionally, can be provided a selection containing interrelation profile subset, to return distinct
        features within that subset.

        :param selection: a subset DataFrame for the interrelation profile, to narrow the scope. Optional, default None.
        :return: feature names as a set of strings
        """
        if selection is None:
            selection = self.df
        features = set(selection.index.unique(level='feature1'))
        features.update(set(selection.index.unique(level='feature2')))
        return features

    def iterate_feature_interrelations(self):
        """Yields all interrelations values between features in the profile, in a tuple.
        Omits self-relations of features.
        Includes imputed values.

        :return: yields tuples of (feature1, feature2, interrelation_value)
        """
        for f1, f2 in self.features2cooccurrences(self.distinct_features(), omit_self_relations=True):
            yield f1, f2, self.interrelation_value(f1, f2)

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

    def standard_raw_interrelation_deviation(self):
        """Provides standard deviation from all explicit (i.e. non-imputed) interrelation values within the profile.
        Ignores self-relations.

        :return: The standard deviation as a float
        """
        explicit_values = self.select_raw_interrelations()['value']
        return float(explicit_values.std())

    @abstractmethod
    def mean_interrelation_value(self):
        """Provides mean value of all interrelation values within the profile, including imputed values.
        Implemented individually within different FeatureInterrelation types, due to imputation differences.
        """
        raise NotImplementedError

    def mean_self_relation_value(self):
        """Provides mean value of all self-relation values within the profile.
        Ignores interrelations.

        :return: the mean explicit interrelation value as a float
        """
        return self.select_self_relations()['value'].mean()

    def mean_raw_interrelation_value(self):
        """Provides mean value of all explicit interrelation values within the profile, i.e. not counting in
        the imputation values.
        Ignores self-relations.

        :return: the mean explicit interrelation value as a float
        """
        return self.select_raw_interrelations()['value'].mean()

    def to_explicit_matrix(self, selection=None):
        """Transforms the interrelation profile, or its subset provided by the selection, into an explicit
        square matrix of interrelation values. Without a specified selection, the entire interrelation profile
        is converted.

        :param selection: an optional subset to form the explicit matrix on. Default None.
        :return: the explicit interrelation table as a DataFrame
        """
        if selection is None:
            selection = self.select_all()
        features = sorted(self.distinct_features(selection))
        dataframe = pandas.DataFrame(index=features, columns=features)
        for feature_a in features:
            for feature_b in features:
                dataframe.at[feature_a, feature_b] = self.interrelation_value(feature_a, feature_b)
        return dataframe

    def to_distance_matrix(self, selection=None, *, distance_conversion_function=None, zero_self_relations=True):
        """Transforms the interrelation profile, or its subset provided by the selection, into an explicit
        distance matrix based on interrelation values. Without a specified selection, the entire interrelation profile
        is converted.

        :param selection: an optional subset to form the explicit matrix on. Default None.
        :param distance_conversion_function: f(x) to transform the interrelation value x into distance. Default 1/x+1.
        :param zero_self_relations: turns distances of all features to themselves into explicit 0. Default True.
        :return: the explicit interrelation table as a DataFrame
        """
        if selection is None:
            selection = self.select_raw_interrelations()
        explicit_matrix = self.to_explicit_matrix(selection)
        if not distance_conversion_function:
            def distance_conversion_function(x):
                return 1 / (x + 1)
        explicit_matrix = explicit_matrix.applymap(distance_conversion_function)
        if zero_self_relations:
            for feature in explicit_matrix.index.values:
                explicit_matrix.at[feature, feature] = 0
        return explicit_matrix

    def to_csv(self, target_file=None):
        """Export the interrelation matrix to a CSV file

        :param target_file: the path or buffer to the export. Default None
        :return: None or string
        """
        return self.df.to_csv(target_file)

    @abstractmethod
    def get_imputation_value(self, f1, f2):
        """Provides the imputation value for feature pair that does not occur within the interrelation profile.
        Implemented individually within different FeatureInterrelation types, due to imputation differences.
        """
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
        if 'imputation_value' not in kwargs:
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

    def get_imputation_value(self, *args):
        """Returns interrelation imputation value. For co-occurrences, this is flat 0, unless z-scored.

        :return: imputation value
        """
        return self.attrs['imputation_value']

    def mean_interrelation_value(self):
        """Provides mean value of all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the mean interrelation value as a float
        """
        raw_interrelations_sum = self.select_raw_interrelations()['value'].sum()
        return raw_interrelations_sum / self.num_max_interrelations()

    def standard_interrelation_deviation(self):
        """Provides standard deviation for all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the standard deviation value of interrelations as a float
        """
        max_interrelations = self.num_max_interrelations()
        raw_interrelations = self.select_raw_interrelations()
        num_imputations = max_interrelations - raw_interrelations.count()
        mean = self.mean_interrelation_value()
        sum_raw_squared_differences = raw_interrelations['value'].apply(lambda x: (x - mean) ** 2).sum()
        sum_imputed_squared_differences = (mean ** 2) * num_imputations
        standard_deviation = numpy.sqrt((sum_raw_squared_differences + sum_imputed_squared_differences)
                                        / max_interrelations)
        return float(standard_deviation)

    def add_another_cooccurrence_profile(self, other):
        """Adds the contents of another co-occurrence profile to this one, in full outer join fashion.
        The addition is done inplace, i.e. on this instance.

        :param other: another CooccurrenceProfile instance
        :return: self
        """
        if not isinstance(other, CooccurrenceProfile):
            raise ValueError(f"Only other CooccurrenceProfile instance can be added to this one, not {type(other)}")
        self.df = self.df.add(other.df, fill_value=0)
        self.df['value'] = self.df['value'].astype(int)
        self.attrs['vector_count'] += other.attrs['vector_count']
        return self

    def __add__(self, other):
        return self.add_another_cooccurrence_profile(other)


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
        df = cooccurrence_profile.df.copy(deep=True)
        df['value'] = df['value'].divide(vector_count)
        kwargs['vector_count'] = vector_count
        kwargs['imputation_probability'] = 1.0 / (vector_count + 1)  # "most-optimist" imputation value
        return cls(df, *args, **kwargs)

    def imputable_standalone_probabilities(self, *args, vector_count=None):
        """Calculates standalone probabilities for individual features, to be used in the "most-optimistic" imputation
        scheme, i.e. presuming that the n+1 feature vector will contain all observed features co-occurring.

        :param vector_count: explicit count of feature vectors, i.e. samples, to optionally manually adjust the probabilities
        :return: a dictionary of feature:probability
        """
        standalone_probabilities = self.self_relations_dict()
        if not vector_count:
            vector_count = self.attrs['vector_count']
        feature2imputable_standalone_probability = {
            feature: (feature_probability * vector_count + 1) / (vector_count + 1)
            for feature, feature_probability in standalone_probabilities.items()}
        return feature2imputable_standalone_probability

    def mean_interrelation_value(self):
        """Provides mean value of all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the mean interrelation value as a float
        """
        raw_interrelations_sum = self.select_raw_interrelations()['value'].sum()
        max_interrelations = self.num_max_interrelations()
        raw_interrelations = self.select_raw_interrelations()
        num_imputations = max_interrelations - raw_interrelations.count()
        imputed_interrelations_sum = self.attrs['imputation_probability'] * num_imputations
        combined_interrelations_sum = raw_interrelations_sum + imputed_interrelations_sum
        return float(combined_interrelations_sum / max_interrelations)

    def standard_interrelation_deviation(self):
        """Provides standard deviation for all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the standard deviation value of interrelations as a float
        """
        max_interrelations = self.num_max_interrelations()
        raw_interrelations = self.select_raw_interrelations()
        num_imputations = max_interrelations - raw_interrelations.count()
        mean = self.mean_interrelation_value()
        sum_raw_squared_differences = raw_interrelations['value'].apply(lambda x: (x - mean) ** 2).sum()
        sum_imputed_squared_differences = ((self.attrs['imputation_probability'] - mean) ** 2) * num_imputations
        standard_deviation = numpy.sqrt((sum_raw_squared_differences + sum_imputed_squared_differences)
                                        / max_interrelations)
        return float(standard_deviation)

    def get_imputation_value(self, *args):
        """Interrelation probability imputation based on "most-optimistic" scenario that the
        co-occurrence would happen in the n+1 sample.

        :return: the imputation value as a float
        """
        return float(self.attrs['imputation_probability'])


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
        if not kwargs.get('vector_count', None):
            vector_count = cooccurrence_probability_profile.attrs['vector_count']
            if not vector_count:
                raise ValueError("If 'vector_count' attribute is not present in the used co-occurrence probability attrs,"
                                 "please either set the co-occurrence probability profile attrs['vector_count'],"
                                 "or add vector_count as a keyword argument to this function")
            kwargs['vector_count'] = vector_count
        kwargs['imputation_probability'] = cooccurrence_probability_profile.select_self_relations()['value'].min()
        imputable_standalone_probabilities = cooccurrence_probability_profile.imputable_standalone_probabilities(
            vector_count=kwargs['vector_count'])
        kwargs['imputation_standalone_probabilities'] = imputable_standalone_probabilities
        standalone_probabilities = cooccurrence_probability_profile.self_relations_dict()
        # TODO: figure how to do this without affecting other optional columns
        df = cooccurrence_probability_profile.df.apply(
            lambda x: numpy.log2(x.value / (standalone_probabilities[x.name[0]] * standalone_probabilities[x.name[1]]))
            if x.name[0] != x.name[1] else x * 0,
            axis=1)  # the if/else clause because P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        return cls(df, *args, **kwargs)

    def get_imputation_value(self, feature1, feature2):
        """PMI imputation is based on the assumption that two of least occurring features within the set can be expected
        to have no interrelation, i.e. their PMI value would be 0. For the more frequently occurring features, their
        lack of co-occurrence is a correspondingly larger surprise, i.e. the imputed PMI values go into the negatives,
        meaning that the feature co-occur less than what could be expected from their individual occurrence
        probabilities, if they were independent. The computation of imputation PMI values (iPMI) for two feature
        is therefore:

        iPMI(base) = log2[(p_least_common_feature * p_least_common_feature)
        / (p_least_common_feature * p_least_common_feature)] = log2[1] = 0

        based on which:

        iPMI(feature1, feature2) = log2[p_least_common_feature**2 / (p_feature1 * p_feature2)]

        where p_least_common_feature is the stand-alone occurrence probability for the least common feature within
        the profile, and p_feature1 and p_feature2 are the stand-alone occurrence probabilities for the features
        which PMI is being imputed.

        :param feature1: First feature for PMI imputation
        :param feature2: Second feature for PMI imputation
        :return: The pair PMI based on imputed values
        """
        if feature1 == feature2:
            return 0  # P(A AND A) = P(A), not P(A)*P(A). And log2(P(A)/P(A)) = log2(1) = 0
        generic_imputation_probability = self.attrs['imputation_probability']
        imputable_marginals = self.attrs['imputation_standalone_probabilities']
        feature1_imputation_probability = imputable_marginals.get(feature1, generic_imputation_probability)
        feature2_imputation_probability = imputable_marginals.get(feature2, generic_imputation_probability)
        return numpy.log2(generic_imputation_probability**2
                          / (feature1_imputation_probability * feature2_imputation_probability))

    def mean_interrelation_value(self):
        """Provides mean value of all PMI values within the profile, including imputed values.
        Ignores self-relations.

        :return: the mean PMI value as a float
        """
        pmi_sum = 0.0
        num_interrelations = 0
        for f1, f2, value in self.iterate_feature_interrelations():
            pmi_sum += value
            num_interrelations += 1
        return pmi_sum / num_interrelations

    def standard_interrelation_deviation(self):
        """Provides standard deviation for all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the standard deviation value of interrelations as a float
        """
        mean = self.mean_interrelation_value()
        sum_raw_squared_differences = 0
        count_interrelations = 0
        for f1, f2, value in self.iterate_feature_interrelations():
            sum_raw_squared_differences += (value - mean) ** 2
            count_interrelations += 1
        standard_deviation = numpy.sqrt((sum_raw_squared_differences / count_interrelations))
        return float(standard_deviation)

    def relative_feature_tightness(self, features):
        """Provides relative feature tightness (RFT) measure for a given set of features.
        RFT quantifies how well do the feature co-occurrence combination in the provided feature vector match
        the interrelations within this reference (Z)PMI profile.

        :param features: an iterable of features
        :return: Feature tightness value as a float
        """
        return self.mean_feature_interrelation_value(features, omit_self_relations=True)


class PointwiseKLDivergenceProfile(InterrelationProfile):
    """An interrelation profile consisting of pointwise Kullback–Leibler divergence values, a measure of statistical
    distances for each feature pair, between its observed co-occurrence in the characterized set and its observed
    co-occurrence in a reference set.
    """

    @classmethod
    def from_cooccurrence_probability_profiles(cls, cooccurrence_probability_profile, reference_probability_profile,
                                               *args, **kwargs):
        """Creates a pointwise KL Divergence interrelation profile quantifying how well do the co-occurrence
        probabilities in the given interrelation profile match those in the given reference interrelation profile.

        pKLD(F1|F2) = log2( P(F1|F2) / Q(F1|F2) )

        where F1 and F2 are observed features, P(F1|F2) is their co-occurrence probability within the evaluated
        interrelation profile, and Q(F1|F2) is the same within the reference interrelation profile.

        :param cooccurrence_probability_profile: the CooccurrenceProbabilityProfile instance to be evaluated
        :param reference_probability_profile: the CooccurrenceProbabilityProfile instance to serve as a reference
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: PointwiseKLDivergenceProfile instance
        """
        imputation_probability_main = cooccurrence_probability_profile.get_imputation_value()
        imputation_probability_ref = reference_probability_profile.get_imputation_value()
        kwargs['imputation_value'] = numpy.log2(imputation_probability_main / imputation_probability_ref)
        df = pandas.merge(cooccurrence_probability_profile.df, reference_probability_profile.df,
                          on=('feature1', 'feature2'), how='outer', suffixes=('_main', '_reference'))
        df = pandas.DataFrame(data=numpy.log2(df['value_main'].fillna(imputation_probability_main)
                                              / df['value_reference'].fillna(imputation_probability_ref)),
                              columns=['value'])
        return cls(df, *args, **kwargs)

    def relative_feature_divergence(self, features):
        """Provides relative feature divergence, i.e. relative feature tightness (RFT) measure for a given set of
        features, against this pointwise KL divergence profile between two interrelation profiles.
        The value quantifies how much does the feature co-occurrence combination in the provided feature vector match
        the interrelations prevalent in the source profile (positive values) compared to those more prevalent
        in the reference profile (negative values).

        :param features: an iterable of features
        :return: Feature divergence value as a float
        """
        return self.mean_feature_interrelation_value(features, omit_self_relations=True)

    def get_imputation_value(self, *args):
        """Pointwise KL imputation for the case that the features do not co-occur in neither the evaluated, nor the
        reference interrelation profile. It is based on the imputation probability for the individual feature profiles.

        :return: Imputation pointwise KLD for feature co-occurrence appearing in neither profile
        """
        return self.attrs['imputation_value']

    def mean_interrelation_value(self):
        """Provides mean value of all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the mean interrelation value as a float
        """
        raw_interrelations_sum = self.select_raw_interrelations()['value'].sum()
        max_interrelations = self.num_max_interrelations()
        num_imputations = max_interrelations - self.select_raw_interrelations().count()
        imputed_interrelations_sum = self.attrs['imputation_value'] * num_imputations
        combined_interrelations_sum = raw_interrelations_sum + imputed_interrelations_sum
        return float(combined_interrelations_sum / max_interrelations)

    def standard_interrelation_deviation(self):
        """Provides standard deviation for all interrelation values within the profile, including imputed values.
        Ignores self-relations.

        :return: the standard deviation value of interrelations as a float
        """
        raw_interrelations = self.select_raw_interrelations()
        max_interrelations = self.num_max_interrelations()
        num_imputations = max_interrelations - raw_interrelations.count()
        mean = self.mean_interrelation_value()
        sum_raw_squared_differences = raw_interrelations['value'].apply(lambda x: (x - mean) ** 2).sum()
        sum_imputed_squared_differences = ((self.attrs['imputation_value'] - mean) ** 2) * num_imputations
        standard_deviation = numpy.sqrt((sum_raw_squared_differences + sum_imputed_squared_differences)
                                        / max_interrelations)
        return float(standard_deviation)


class PointwiseJeffreysDivergenceProfile(PointwiseKLDivergenceProfile):
    """An interrelation profile consisting of pointwise Jeffreys divergence values, a measure of statistical
    distances for each feature pair, between its observed co-occurrence in the characterized set and its observed
    co-occurrence in a reference set - and vice versa.
    A symmetric variant of the KL divergence.
    """

    @classmethod
    def from_cooccurrence_probability_profiles(cls, cooccurrence_probability_profile, reference_probability_profile,
                                               *args, **kwargs):
        """Creates a pointwise Jeffreys divergence interrelation profile quantifying how well do the
        co-occurrence probabilities in the given interrelation profile match those in the given reference interrelation
        profile.

        pJD(F1|F2) == pJD(F2|F1) = abs(pKLD(F1|F2)) + abs(pKLD(F2|F1))

        where F1 and F2 are observed features, pKLD(F1|F2) is their pointwise KL divergence.

        :param cooccurrence_probability_profile: first CooccurrenceProbabilityProfile instance
        :param reference_probability_profile: second CooccurrenceProbabilityProfile instance
        :param args: any further arguments to be passed to the InterrelationProfile init
        :param kwargs: any further keyword arguments to be passed to the InterrelationProfile init
        :return: PointwiseJeffreysDivergenceProfile instance
        """
        imputation_probability_main = cooccurrence_probability_profile.get_imputation_value()
        imputation_probability_ref = reference_probability_profile.get_imputation_value()
        kwargs['imputation_value'] = (numpy.log2(imputation_probability_main / imputation_probability_ref)
                                      + numpy.log2(imputation_probability_ref / imputation_probability_main))
        df = pandas.merge(cooccurrence_probability_profile.df, reference_probability_profile.df,
                          on=('feature1', 'feature2'), how='outer', suffixes=('_main', '_reference'))
        main_values = df['value_main'].fillna(imputation_probability_main)
        reference_values = df['value_reference'].fillna(imputation_probability_ref)
        df = pandas.DataFrame(
            data=abs(numpy.log2(main_values / reference_values)) + abs(numpy.log2(reference_values / main_values)),
            columns=['value'])
        return cls(df, *args, **kwargs)
