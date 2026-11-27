# Student Performance Analysis

## Project Overview

This project explores student academic performance using exploratory data analysis (EDA) and machine learning techniques.

The main objective is to investigate relationships between students' demographic, social, and academic background features and their final academic outcomes, and to develop classification models that predict whether a student passes or fails.

## Objectives

- Explore the dataset and understand its structure.
- Investigate relationships between student characteristics and academic performance.
- Identify important features associated with student outcomes.
- Build and evaluate machine learning classification models.
- Compare model performance, particularly in identifying students at risk of failing.

## Dataset

The project uses the Student Performance dataset from the UCI Machine Learning Repository.

- Dataset: Student Performance (Mathematics)
- Number of records: 395
- Number of original features: 33
- Missing values: 0
- Duplicate records: 0

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance

## Target Variable

The target variable is created from the final grade (G3):

- 1: Pass (G3 >= 10)
- 0: Fail (G3 < 10)

The original grade columns G1 and G2 are excluded from the input features to prevent data leakage. G3 is used only to create the target variable and is not included as a predictor.

## Data Analysis

The exploratory analysis examined several relationships:

- Study time and final academic performance.
- Absences and final grades.
- Previous failures and academic outcomes.
- Feature importance using a Decision Tree model.

The results suggest that previous failures are strongly associated with lower final grades in this dataset. However, these relationships do not establish causation.

## Data Preprocessing

The following preprocessing steps were applied:

- Train-test split (80/20) with stratification.
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- ColumnTransformer for combining preprocessing steps.
- Pipeline-based preprocessing during cross-validation to prevent data leakage.

The processed dataset contains 56 features.

## Machine Learning Models

The following models were evaluated:

1. Dummy Classifier (Baseline)
2. Logistic Regression
3. Balanced Logistic Regression
4. Decision Tree Classifier
5. Tuned Decision Tree using GridSearchCV

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Stratified 5-fold Cross-Validation

Special attention was given to recall for the Fail class because identifying students at risk of failing is an important objective.

### Test Set Results

| Model | Accuracy | Fail Recall | Fail Precision | Fail F1 |
|---|---:|---:|---:|---:|
| Dummy Classifier (Baseline) | 67.09% | 0.00% | — | 0.00% |
| Logistic Regression | 65.82% | 34.62% | 47.37% | 40.00% |
| Balanced Logistic Regression | 59.49% | 46.15% | 40.00% | 42.86% |
| Tuned Decision Tree | 65.82% | 34.62% | 47.37% | 40.00% |

The Dummy Classifier achieved 67.09% accuracy by predicting every student as Pass. However, it failed to identify any students who actually failed. This demonstrates why accuracy alone can be misleading when evaluating imbalanced classification problems.

The Balanced Logistic Regression model achieved the highest recall for the Fail class, identifying more students who actually failed.

However, this improvement came with a lower overall accuracy and more false positive predictions.

The Logistic Regression and Tuned Decision Tree models produced the same confusion matrix, and therefore the same test metrics, although their predictions differ for 6 of the 79 test students.

## Visualizations

Running the script saves these figures to the `figures/` folder and displays them:

| File | Content |
|---|---|
| `grade_by_study_time.png` | Mean final grade by weekly study time |
| `absences_vs_grade.png` | Absences compared with final grade |
| `grade_by_failures.png` | Mean final grade by number of previous failures |
| `feature_importance.png` | Top 10 features of the Decision Tree (depth 4) |
| `decision_tree.png` | First two levels of the Decision Tree (depth 4) |
| `confusion_matrices.png` | Test set confusion matrices of the compared models |
| `model_comparison.png` | Test set accuracy and Fail-class metrics |

## Limitations

- The dataset contains only 395 observations.
- The results may not generalize to other schools or populations.
- The models have limited ability to identify students who fail.
- Feature relationships indicate associations, not causal effects.
- The evaluation is based on a single stratified train-test split.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Git and GitHub

## Project Structure

```text
student-performance-analysis/
├── data/
│   └── student-mat.csv
├── figures/              # created by the script
├── notebooks/
│   ├── student_analysis.ipynb
│   └── test.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/esm4aeil/student-performance-analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Python analysis script:

```bash
python notebooks/test.py
```

## Conclusion

This project demonstrates a complete introductory machine learning workflow, including data exploration, preprocessing, model development, evaluation, and error analysis.

The findings highlight the importance of evaluating classification models beyond accuracy, particularly when identifying students at risk of failing.

This project is an educational and exploratory study, not a validated student assessment system.