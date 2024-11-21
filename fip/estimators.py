from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np

from fip.profiles import CooccurrenceProfile, CooccurrenceProbabilityProfile, PointwiseKLDivergenceProfile


class PKLDivergenceEstimator(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible estimator that wraps the pointwise Kullback-Leibler divergence variant
    of the FIP methodology
    """

    def __init__(self):
        self.pkld_profile = None
        self.classification_cutoff = 0.0
        self.X_ = None
        self.y_ = None
        self.classes_ = [0, 1]

    @staticmethod
    def determine_classification_cutoff(labels, pkld_profile) -> float:
        """
        The default classification cutoff is the n-th percentile corresponding
        to the ratio of positive to negative class labels in the training data.

        :param labels: list of labels
        :param pkld_profile: PointwiseKLDivergenceProfile instance
        :return: float
        """
        positive_cutoff_percentile = (sum(labels) / len(labels)) * 100
        hit_interrelation_values = pkld_profile.features_interrelation_values(
            pkld_profile.distinct_features(), omit_self_relations=True)
        hit_interrelation_values = np.array(hit_interrelation_values)
        classification_cutoff = np.percentile(hit_interrelation_values, positive_cutoff_percentile)
        return classification_cutoff

    def fit(self, X, y) -> object:
        """
        Fit the estimator to given data.

        :param X: list of lists of features
        :param y: list of labels
        :return: PKLDivergenceEstimator
        """
        # X, y = check_X_y(X, y, accept_sparse=True)  # looks like check_X_y hard rejects X members of variable length
        positive_cooccurrence_profile = CooccurrenceProfile.from_feature_lists(
            (flist for flist, label in zip(X, y) if label))
        positive_cooccurrence_probability_profile = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            positive_cooccurrence_profile)
        del positive_cooccurrence_profile
        negative_cooccurrence_profile = CooccurrenceProfile.from_feature_lists(
            (flist for flist, label in zip(X, y) if not label))
        negative_cooccurrence_probability_profile = CooccurrenceProbabilityProfile.from_cooccurrence_profile(
            negative_cooccurrence_profile)
        del negative_cooccurrence_profile
        self.pkld_profile = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            positive_cooccurrence_probability_profile, negative_cooccurrence_probability_profile)
        training_set_scores = [self.pkld_profile.relative_feature_divergence(flist) for flist in X]
        positive_cutoff_percentile = (sum(y) / len(y)) * 100
        self.classification_cutoff = np.percentile(training_set_scores, positive_cutoff_percentile)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X) -> list:
        """
        Predict class labels for given data.

        :param X: list of feature sets
        :return: list of predicted class labels (1, 0)
        """
        return [x > self.classification_cutoff for x in self.predict_proba(X)]

    def is_fitted(self) -> bool:
        """
        Check if the estimator has been fitted to data.
        """
        return self.pkld_profile is not None

    def predict_proba(self, X) -> list:
        """
        Predict class scores for given data.

        :param X: list of feature sets
        """
        check_is_fitted(self)
        return [self.pkld_profile.relative_feature_divergence(flist) for flist in X]


class PKLDivergenceMultilabelEstimator(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible estimator that wraps the pointwise Kullback-Leibler divergence variant
    of the FIP methodology for multilabel classification, calculated ad-hoc from cooccurrence
    probability profile.
    """

    def __init__(self):
        self.cooccurrence_p_profile = None
        self.classification_cutoffs = None
        self.X_ = None
        self.y_ = None
        self.classes_ = None

    def fit(self, X, y) -> object:
        """
        Fit the estimator to given data.

        :param X: list of lists of features
        :param y: list of lists of labels
        :return: PKLDivergenceMultilabelEstimator
        """
        self.classes_ = set()
        for labels in y:
            self.classes_.update(labels)
        self.classes_ = list(self.classes_)
        cooccurrence_profile = CooccurrenceProfile.from_feature_lists(
            (set(flist).union(set(labels)) for flist, labels in zip(X, y)), tracked_features=self.classes_
        )
        self.cooccurrence_p_profile = CooccurrenceProbabilityProfile.from_cooccurrence_profile(cooccurrence_profile)
        del cooccurrence_profile
        self.X_ = X
        self.y_ = y
        self.classification_cutoffs = {label: 0 for label in self.classes_}  # for now, might be changed later
        return self

    def predict(self, X) -> list:
        """
        Predict class labels for given data.

        :param X: list of feature sets
        :return: list of predicted class labels
        """
        class2pkld = self.predict_proba(X)
        return [label for label, pkld in class2pkld.items() if pkld > self.classification_cutoffs[label]]

    def predict_proba(self, X) -> dict:
        """
        Predict class scores for given data.

        :param X: list of feature sets
        """
        check_is_fitted(self)
        return self.cooccurrence_p_profile.tracked_features_pkld(X)


    def is_fitted(self) -> bool:
        """
        Check if the estimator has been fitted to data.
        """
        return self.cooccurrence_p_profile is not None