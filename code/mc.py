"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q4. Method comparison.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from data import make_dataset1, make_dataset_breast_cancer

# -------------------------
# Hyperparameter tuning
# -------------------------
def tune_hyperparameter(method, X_train, y_train, candidate_values):
    best_value = None
    best_acc = -np.inf

    for val in candidate_values:
        if method == "dt":
            clf = DecisionTreeClassifier(max_depth=val, random_state=0)
        elif method == "knn":
            if val > len(X_train):
                continue  
            clf = KNeighborsClassifier(n_neighbors=val)
        else:
            raise ValueError("Unknown method")

        clf.fit(X_train, y_train)
        acc = accuracy_score(y_train, clf.predict(X_train))
        if acc > best_acc:
            best_acc = acc
            best_value = val

    return best_value

# -------------------------
# Evaluate all methods
# -------------------------
def evaluate_methods(X, y):
    results = {}

    # Split once into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    # Tune hyperparameters using only training set 
    dt_depths = [1, 2, 4, 6, None]
    knn_neighbors = [1, 5, 25, 125, 500]

    best_depth = tune_hyperparameter("dt", X_train, y_train, dt_depths)
    best_k = tune_hyperparameter("knn", X_train, y_train, knn_neighbors)

    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=0)
    dt_clf.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_clf.predict(X_test))

    # kNN
    knn_clf = KNeighborsClassifier(n_neighbors=best_k)
    knn_clf.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn_clf.predict(X_test))

    # LDA
    lda_clf = LinearDiscriminantAnalysis()
    lda_clf.fit(X_train, y_train)
    lda_acc = accuracy_score(y_test, lda_clf.predict(X_test))

    # QDA
    qda_clf = QuadraticDiscriminantAnalysis()
    qda_clf.fit(X_train, y_train)
    qda_acc = accuracy_score(y_test, qda_clf.predict(X_test))

    results["DT"] = dt_acc
    results["kNN"] = knn_acc
    results["LDA"] = lda_acc
    results["QDA"] = qda_acc
    results["best_depth"] = best_depth
    results["best_k"] = best_k

    return results

# -------------------------
# Run on both datasets Dataset1 and BreastCancer
# -------------------------
if __name__ == "__main__":
    datasets = {
        "Dataset1": make_dataset1(1200, random_state=0),
        "BreastCancer": make_dataset_breast_cancer(random_state=0)
    }

    for name, (X, y) in datasets.items():
        res = evaluate_methods(X, y)
        print(f"{name} results:")
        print(f"DT (max_depth={res['best_depth']}): {res['DT']:.3f}")
        print(f"kNN (n_neighbors={res['best_k']}): {res['kNN']:.3f}")
        print(f"LDA: {res['LDA']:.3f}")
        print(f"QDA: {res['QDA']:.3f}")
        print("-" * 40)
