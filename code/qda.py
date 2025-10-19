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

        # Compute priors, means, and covariances
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.covs_ = []

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = X_c.shape[0] / X.shape[0]
            self.means_[i] = np.mean(X_c, axis=0)
            self.covs_.append(np.cov(X_c, rowvar=False))

        if self.lda:
            # Shared covariance matrix (pooled)
            pooled_cov = np.zeros_like(self.covs_[0])
            for i, c in enumerate(self.classes_):
                n_c = np.sum(y == c)
                pooled_cov += (n_c - 1) * self.covs_[i]
            pooled_cov /= (X.shape[0] - n_classes)
            self.pooled_cov_ = pooled_cov
            self.inv_pooled_cov_ = np.linalg.inv(pooled_cov)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes_):
            mean = self.means_[i]
            if self.lda:
                inv_cov = self.inv_pooled_cov_
                scores[:, i] = (
                    X @ inv_cov @ mean
                    - 0.5 * mean.T @ inv_cov @ mean
                    + np.log(self.priors_[i])
                )
            else:
                cov = self.covs_[i]
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                diff = X - mean
                scores[:, i] = (
                    -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                    - 0.5 * np.log(det_cov)
                    + np.log(self.priors_[i])
                )

        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        scores = np.zeros((n_samples, n_classes))

        for i, c in enumerate(self.classes_):
            mean = self.means_[i]
            if self.lda:
                inv_cov = self.inv_pooled_cov_
                scores[:, i] = (
                    X @ inv_cov @ mean
                    - 0.5 * mean.T @ inv_cov @ mean
                    + np.log(self.priors_[i])
                )
            else:
                cov = self.covs_[i]
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
                diff = X - mean
                scores[:, i] = (
                    -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                    - 0.5 * np.log(det_cov)
                    + np.log(self.priors_[i])
                )

        # Convert discriminant scores to probabilities
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=1, keepdims=True)
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
