"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
2025-2026

Q1. Decision Trees.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from data import make_dataset1
from plot import plot_boundary


# Evaluate Decision Tree
def evaluate_decision_tree(X_train, y_train, X_test, y_test, max_depth):
    """Train and evaluate a Decision Tree for a given max_depth."""
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return clf, acc

# Experiment with Decision Tree
def experiment_decision_tree(n_runs=5, n_points=1200, depths=[1, 2, 4, 6, None]):
    """Run multiple experiments with different random seeds and tree depths."""
    results = {d: [] for d in depths}

    for seed in range(n_runs):
        # Generate dataset and split (75% train / 25% test)
        X, y = make_dataset1(n_points, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

        for d in depths:
            clf, acc = evaluate_decision_tree(X_train, y_train, X_test, y_test, d)
            results[d].append(acc)

            # Plot boundary (on an independent dataset for visualization)
            X_plot, y_plot = make_dataset1(n_points, random_state=seed + 100)
            plot_boundary(
                fname=f"Dt_Plots/dt_depth_{d}_run_{seed}",
                fitted_estimator=clf,
                X=X_plot,
                y=y_plot,
                title=f"Decision Tree (depth={d}) - run {seed}",
            )

    # Calculating mean ± std
    summary = {}
    print("\n=== Decision Tree Results (5 seeds) ===")
    for d in depths:
        mean_acc = np.mean(results[d])
        std_acc = np.std(results[d])
        summary[d] = (mean_acc, std_acc)
        print(f"max_depth={d}: {mean_acc:.3f} ± {std_acc:.3f}")

    return summary


if __name__ == "__main__":
    summary = experiment_decision_tree()
