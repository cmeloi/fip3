from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

from fip.profiles import CooccurrenceProfile, PointwiseKLDivergenceProfile


class PKLDivergenceEstimator(BaseEstimator, ClassifierMixin):
    """A scikit-learn compatible estimator that wraps the pointwise Kullback-Leibler divergence variant
    of the FIP methodology
    """

    def __init__(self):
        self.pkld_profile = None
        self.classification_cutoff = 0.0

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        positive_cooccurrence_profile = CooccurrenceProfile.from_feature_lists(
            (flist for flist, label in zip(X, y) if label))
        negative_cooccurrence_profile = CooccurrenceProfile.from_feature_lists(
            (flist for flist, label in zip(X, y) if not label))
        self.pkld_profile = PointwiseKLDivergenceProfile.from_cooccurrence_probability_profiles(
            positive_cooccurrence_profile, negative_cooccurrence_profile)
        # TODO: add a way to determine the optimal threshold between positive and negative
        #  on these usually lognormal distributions
        return self

    def predict(self, X):
        return [x > self.classification_cutoff for x in self.predict_proba(X)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return [self.pkld_profile.mean_feature_interrelation_values(flist, omit_self_relations=True) for flist in X]

