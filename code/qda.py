"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q3. Linear/Quadratic Discriminant Analysis (LDA/QDA).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import make_dataset1, make_dataset_breast_cancer


# ==========================================
# Definition of LDA / QDA Classifier
# ==========================================
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

        # If LDA → shared covariance matrix (pooled)
        if self.lda:
            pooled_cov = np.zeros_like(self.covs_[0])
            for i, c in enumerate(self.classes_):
                n_c = np.sum(y == c)
                pooled_cov += (n_c - 1) * self.covs_[i]
            pooled_cov /= (X.shape[0] - n_classes)
            self.pooled_cov_ = pooled_cov
            self.inv_pooled_cov_ = np.linalg.inv(pooled_cov)

        return self

    def predict(self, X):
        """Predict class labels."""
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
        """Predict class probabilities."""
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

        # Convert scores to probabilities
        scores -= np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs


# ==========================================
# Q3.2 - Decision Boundaries Visualization
# ==========================================
if __name__ == "__main__":
    X, y = make_dataset1(1200, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Meshgrid for decision surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Draw both LDA and QDA figures and save them
    for mode, title in [(True, "LDA"), (False, "QDA")]:
        model = QuadraticDiscriminantAnalysis().fit(X_train, y_train, lda=mode)
        Z = model.predict(grid).reshape(xx.shape)

        plt.figure()
        plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap="coolwarm", edgecolor="k")
        plt.title(f"{title} Decision Boundary on Dataset 1")
        plt.xlabel("x₀")
        plt.ylabel("x₁")
        plt.tight_layout()
        plt.savefig(f"{title}_Decision_Boundary.png", dpi=300)
        plt.close()  # Prevent freeze

    # ==========================================
    # Q3.3 - Average Accuracy over 5 Runs
    # ==========================================
    def evaluate_dataset(make_data_fn, name):
        lda_acc, qda_acc = [], []

        for seed in range(5):
            X, y = make_data_fn(1200 if name == "Dataset 1" else None, random_state=seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

            for mode, acc_list in [(True, lda_acc), (False, qda_acc)]:
                model = QuadraticDiscriminantAnalysis().fit(X_train, y_train, lda=mode)
                preds = model.predict(X_test)
                acc_list.append(accuracy_score(y_test, preds))

        print(f"\n{name}")
        print(f"LDA: Mean Accuracy = {np.mean(lda_acc):.3f}, Std = {np.std(lda_acc):.3f}")
        print(f"QDA: Mean Accuracy = {np.mean(qda_acc):.3f}, Std = {np.std(qda_acc):.3f}")

    # Run evaluation on both datasets
    evaluate_dataset(make_dataset1, "Dataset 1")
    evaluate_dataset(make_dataset_breast_cancer, "Breast Cancer")