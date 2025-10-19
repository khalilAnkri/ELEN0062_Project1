"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q3. Linear/Quadratic Discriminant Analysis (LDA/QDA).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def fit(self, X, y, lda=False):
        """Fit a linear (LDA) or quadratic (QDA) discriminant analysis model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Compute priors π_k
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covs_ = []

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = X_c.shape[0] / X.shape[0]
            self.means_[i] = X_c.mean(axis=0)
            cov_c = np.cov(X_c, rowvar=False)
            self.covs_.append(cov_c)

        if lda:
            # Shared covariance for LDA: weighted average of class covariances
            pooled_cov = np.zeros_like(self.covs_[0])
            for i, c in enumerate(self.classes_):
                n_c = np.sum(y == c)
                pooled_cov += (n_c - 1) * self.covs_[i]
            pooled_cov /= (X.shape[0] - n_classes)
            self.pooled_cov_ = pooled_cov
            self.inv_pooled_cov_ = np.linalg.inv(pooled_cov)

        return self

    def _discriminant_function(self, X, class_idx):
        """Compute log posterior up to additive constant for class k."""
        mean = self.means_[class_idx]
        if self.lda:
            # Linear discriminant: shared covariance
            inv_cov = self.inv_pooled_cov_
            term = X @ inv_cov @ mean - 0.5 * mean.T @ inv_cov @ mean + np.log(
                self.priors_[class_idx]
            )
            return term
        else:
            # Quadratic discriminant: class-specific covariance
            cov = self.covs_[class_idx]
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            diff = X - mean
            term = (
                -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                - 0.5 * np.log(det_cov)
                + np.log(self.priors_[class_idx])
            )
            return term

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [self._discriminant_function(X, i) for i in range(len(self.classes_))]
        )
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.column_stack(
            [self._discriminant_function(X, i) for i in range(len(self.classes_))]
        )
        # Softmax for probabilities
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


if __name__ == "__main__":
    # Small test example
    from data import make_dataset1
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_dataset1(1200, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    for mode in [False, True]:
        model = QuadraticDiscriminantAnalysis().fit(X_train, y_train, lda=mode)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{'LDA' if mode else 'QDA'} accuracy: {acc:.3f}")
