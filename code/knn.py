"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q2. k-Nearest Neighbors (kNN).
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset1
from plot import plot_boundary


def evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors):
    """Train and evaluate a KNN classifier for a given number of neighbors."""
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc


def experiment_knn(
    n_runs=5, n_points=1200, neighbors_list=[1, 5, 25, 125, 500, 899]
):
    """Run multiple experiments for various k values and seeds."""
    results = {k: [] for k in neighbors_list}

    # Creating the output folder
    os.makedirs("KNN_Plots", exist_ok=True)

    for seed in range(n_runs):
        # Generating dataset and split 75/25 (train/test)
        X, y = make_dataset1(n_points, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

        for k in neighbors_list:
            clf, acc = evaluate_knn(X_train, y_train, X_test, y_test, k)
            results[k].append(acc)

            # dataset for boundary visualization
            X_plot, y_plot = make_dataset1(n_points, random_state=seed + 100)
            plot_boundary(
                fname=f"KNN_Plots/knn_k_{k}_run_{seed}",
                fitted_estimator=clf,
                X=X_plot,
                y=y_plot,
                title=f"kNN (k={k}) - run {seed}",
            )

    # calculating mean ± std across seeds
    summary = {}
    print("\n=== kNN Results (5 seeds) ===")
    for k in neighbors_list:
        mean_acc = np.mean(results[k])
        std_acc = np.std(results[k])
        summary[k] = (mean_acc, std_acc)
        print(f"k={k}: {mean_acc:.3f} ± {std_acc:.3f}")

    return summary


if __name__ == "__main__":
    summary = experiment_knn()
