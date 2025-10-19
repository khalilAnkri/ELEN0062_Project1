"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q4. Method comparison.
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset1, make_dataset_breast_cancer
from qda import QuadraticDiscriminantAnalysis


def tune_and_evaluate(clf, param_grid, X_train, y_train, X_test, y_test):
    """Tune hyperparameters via CV and evaluate on test set."""
    grid = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, grid.best_params_


def run_experiments(make_dataset, n_points=None, n_runs=5, dataset_name="synthetic"):
    """Compare DT, kNN, LDA, QDA on given dataset."""
    print(f"\n=== Dataset: {dataset_name} ===")

    methods = ["DecisionTree", "kNN", "LDA", "QDA"]
    results = {m: [] for m in methods}

    for seed in range(n_runs):
        # Generate dataset and split
        X, y = make_dataset(n_points, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

        # --- Decision Tree ---
        acc, best_params = tune_and_evaluate(
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [1, 2, 4, 6, None]},
            X_train,
            y_train,
            X_test,
            y_test,
        )
        results["DecisionTree"].append(acc)

        # --- kNN ---
        acc, best_params = tune_and_evaluate(
            KNeighborsClassifier(),
            {"n_neighbors": [1, 5, 25, 125, 500, 899]},
            X_train,
            y_train,
            X_test,
            y_test,
        )
        results["kNN"].append(acc)

        # --- LDA ---
        lda = QuadraticDiscriminantAnalysis().fit(X_train, y_train, lda=True)
        preds = lda.predict(X_test)
        results["LDA"].append(accuracy_score(y_test, preds))

        # --- QDA ---
        qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train, lda=False)
        preds = qda.predict(X_test)
        results["QDA"].append(accuracy_score(y_test, preds))

    # Compute mean ± std
    summary = {}
    print("\nMethod Comparison (mean ± std over 5 seeds):")
    for m in methods:
        mean_acc = np.mean(results[m])
        std_acc = np.std(results[m])
        summary[m] = (mean_acc, std_acc)
        print(f"{m:>12}: {mean_acc:.3f} ± {std_acc:.3f}")

    return summary


if __name__ == "__main__":
    # Run on both datasets
    run_experiments(make_dataset1, n_points=1200, dataset_name="Synthetic (2D)")
    run_experiments(make_dataset_breast_cancer, dataset_name="Breast Cancer (30D)")
