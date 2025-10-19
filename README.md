# ELEN0062_Project1# ELEN0062 - Project 1: Classification Algorithms

## Introduction

[cite_start]This repository contains the solution for **Project 1** of the **ELEN0062 - Introduction to Machine Learning** course at ULiège[cite: 1]. [cite_start]The project focuses on implementing and analyzing classical classification algorithms, including **Decision Trees (DT)**, **K-Nearest Neighbors (KNN)**, **Quadratic Discriminant Analysis (QDA)**, and **Linear Discriminant Analysis (LDA)**[cite: 1, 3]. [cite_start]A key objective is to observe and discuss concepts such as underfitting, overfitting, and the impact of model complexity on decision boundaries[cite: 3].

## Group Information

| Role | Name | ULiège Email |
| :--- | :--- | :--- |
| Student 1 | Samira ben ahmed | Samira.BenAhmed@student-uliege.be |
| Student 2 | Mohamed-Khalil Ankri | Mohamed-Khalil.Ankri@student.uliege.be |
| Student 3  | Ishahk Hamad | ishak.hamad@student.uliege.be |

## Submission Details

* [cite_start]**Course:** ELEN0062 - Introduction to Machine Learning [cite: 1]
* [cite_start]**Due Date:** October 24th, 23:59 GMT+2 [cite: 7]
* [cite_start]**Submission Platform:** Gradescope [cite: 7, 31]
* [cite_start]**Gradescope Entry Code:** `7XYGVW` [cite: 31]
* [cite_start]**Deliverables:** Separate Python scripts (`.py`) and a detailed report (`report.pdf`)[cite: 4, 6, 8].

## Project Structure

The repository is organized as follows:
ELEN0062_Project1/
├── dt.py           # Decision Tree implementation and experiments
├── knn.py          # K-Nearest Neighbors implementation and experiments
├── qda.py          # Custom QDA/LDA estimator implementation and experiments
├── data.py         # Provided script for dataset generation (make_dataset1) ├── plot.py         # Provided script for plotting decision boundaries ├── report.pdf      # The final project report (max ~4 pages without figures) 

## Setup and Reproducibility

### Prerequisites

You will need a Python environment with the following libraries:

* **Python 3.x**
* **scikit-learn** (`sklearn`) [cite: 21, 50]
* **NumPy**
* **Matplotlib** (likely used by `plot.py`)

You can install the necessary dependencies using `pip`:

```bash
pip install scikit-learn numpy matplotlib
