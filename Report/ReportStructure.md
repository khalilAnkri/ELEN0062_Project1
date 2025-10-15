# ELEN0062 - Project 1: Classification Algorithms Report

## 1. Introduction

### 1.1 Project Objectives
The goal of this project is to get familiar with classical classification algorithms and concepts such as underfitting and overfitting.

### 1.2 Algorithms Studied
The four algorithms studied are:  
- **Decision Tree (DT)**  
- **K-Nearest Neighbors (KNN)**  
- **Quadratic Discriminant Analysis (QDA)**  
- **Linear Discriminant Analysis (LDA)**  

### 1.3 Datasets
- **Dataset 1:** Generated using `make_dataset1`, consisting of two Gaussian distributions with 1,200 samples.  
- **Dataset 2:** Real-world breast cancer classification dataset, with 569 samples and 30 features.  

### 1.4 Data Split
Both datasets use a fixed split of **75% for training** and **25% for testing**.

---

## 2. Decision Tree Analysis (`dt.py`)
*(Addresses Questions 1.1 and 1.2 from the project statement)*

### 2.1 Impact of `max_depth` on Decision Boundary (Q1.1a)
**Decision Boundary Illustration (Figure 2.1):**  
Include 5 plots for `max_depth` ∈ {1, 2, 4, 6, None}. Use a dataset independent of the training set for visualization.

**Explanation:**  
Describe how increasing tree complexity affects the classification boundary.

### 2.2 Underfitting and Overfitting (Q1.1b)
Discuss when the model is **underfitting** (e.g., low `max_depth`) and **overfitting** (e.g., `max_depth=None`) based on plots and accuracy results.

### 2.3 Model Confidence (Q1.1c)
Explain why the model appears more confident when `max_depth` is largest (`None`).

### 2.4 Test Accuracy and Variability (Q1.2)
**Table 2.1:** Average test accuracies and standard deviations over five generations of Dataset 1 for each `max_depth`.  

**Commentary:**  
Briefly comment on the accuracy trends in relation to underfitting and overfitting.

---

## 3. K-Nearest Neighbors Analysis (`knn.py`)
*(Addresses Questions 2.1 and 2.2 from the project statement)*

### 3.1 Impact of `n_neighbors` on Decision Boundary (Q2.1a, Q2.1b)
**Decision Boundary Illustration (Figure 3.1):**  
Include 6 plots for `n_neighbors` ∈ {1, 5, 25, 125, 500, 899} on Dataset 1.

**Commentary:**  
Explain how the decision boundary evolves from noisy to smooth as `n_neighbors` increases.

### 3.2 Test Accuracy and Variability (Q2.2)
**Table 3.1:** Average test accuracies and standard deviations over five generations of Dataset 1 for each `n_neighbors`.

**Commentary:**  
Briefly comment on the results.

---

## 4. Discriminant Analysis (`qda.py`)
*(Addresses Questions 3.1, 3.2, and 3.3 from the project statement)*

### 4.1 Mathematical Derivation (Q3.1)
Show mathematically that the **decision boundary of QDA is quadratic** and the **decision boundary of LDA is linear** in the two-class case.

### 4.2 Decision Boundaries (Q3.2)
**Decision Boundary Illustration (Figure 4.1):**  
Include 2 plots illustrating decision boundaries for both QDA and LDA on Dataset 1.

**Commentary:**  
Briefly comment on the results.

### 4.3 Performance Comparison (Q3.3)
**Table 4.1:** Average accuracy and standard deviation over five generations for both QDA and LDA on Dataset 1 and Dataset 2.

**Commentary:**  
Compare QDA and LDA based on performance.

---

## 5. Method Comparison and Hyperparameter Tuning
*(Addresses Questions 4.1, 4.2, and 4.3 from the project statement)*

### 5.1 Hyperparameter Tuning Strategy (Q4.1)
Explain a method to tune `max_depth` (DT) and `n_neighbors` (KNN) using only the learning set (e.g., cross-validation).

### 5.2 Tuned Performance (Q4.2)
**Table 5.1:** Average test accuracies and standard deviations over five generations for the tuned DT and KNN on both datasets.

### 5.3 Final Ranking and Discussion (Q4.3)
**Table 5.2 (Summary):** Consolidate results for all four tuned methods (DT, KNN, QDA, LDA) on both datasets.

**Discussion:**  
- Compare these results with those obtained in question 3.3.  
- Rank the four methods on the two datasets and discuss differences.

---

## 6. Conclusion
- Summarize key findings regarding model complexity for DT and KNN.  
- Reiterate main conclusions from the final comparison and ranking of all four methods on the two datasets.
