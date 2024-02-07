from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np

from fip.profiles import CooccurrenceProfile, CooccurrenceProbabilityProfile, PointwiseKLDivergenceProfile


class PKLDivergenceEstimator(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that wraps the pointwise Kullback-Leibler divergence variant
    of the FIP methodology
    """

    def __init__(self):
        self.pkld_profile = None
        self.classification_cutoff = 0.0

    @staticmethod
    def determine_classification_cutoff(labels, pkld_profile) -> float:
        """The default classification cutoff is the n-th percentile corresponding
        to the ratio of positive to negative class labels in the training data.

        :param labels: list of labels
        :param pkld_profile: PointwiseKLDivergenceProfile instance
        :return: float
        """
        positive_cutoff_percentile = (sum(labels) / len(labels)) * 100
        classification_cutoff = np.percentile(pkld_profile.interrelation_values(), positive_cutoff_percentile)
        return classification_cutoff

    def fit(self, X, y):
        """Fit the estimator to given data.

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
        self.classification_cutoff = self.determine_classification_cutoff(y, self.pkld_profile)
        return self

    def predict(self, X):
        return [x > self.classification_cutoff for x in self.predict_proba(X)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return [self.pkld_profile.mean_feature_interrelation_values(flist, omit_self_relations=True) for flist in X]

