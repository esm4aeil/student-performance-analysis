import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ==========================================
# 1. Load Dataset
# ==========================================

project_path = Path(__file__).resolve().parent.parent
data_path = project_path / "data" / "student-mat.csv"

df = pd.read_csv(data_path, sep=";")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

# ==========================================
# 2. Data Quality & Statistics
# ==========================================

print("\nFinal Grade Statistics:")
print(df["G3"].describe())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Pass / Fail Statistics

passed = (df["G3"] >= 10).sum()
failed = (df["G3"] < 10).sum()

print("\nPass/Fail Statistics:")
print("Passed:", passed)
print("Failed:", failed)

# ==========================================
# 3. Exploratory Data Analysis (EDA)
# ==========================================

# Study Time Analysis

study_performance = df.groupby("studytime")["G3"].mean()

print("\nAverage Grade by Study Time:")
print(study_performance)

# Absences Correlation

correlation = df["absences"].corr(df["G3"])

print("\nCorrelation between Absences and G3:")
print(correlation)

# Previous Failures Analysis

failure_analysis = df.groupby("failures")["G3"].agg(
    ["count", "mean", "median"]
)

print("\nFailure Analysis:")
print(failure_analysis)

# ==========================================
# 4. Create Target Variable
# ==========================================

# 1 = Pass
# 0 = Fail

y = (df["G3"] >= 10).astype(int)

# ==========================================
# 5. Select Features
# ==========================================

# Remove G1, G2 and G3 to avoid grade leakage

X = df.drop(columns=["G1", "G2", "G3"])

# ==========================================
# 6. Train-Test Split
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# ==========================================
# 7. Data Preprocessing
# ==========================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns

print("\nNumerical Features:", len(numeric_features))
print("Categorical Features:", len(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Results:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)

# ==========================================
# 8. Baseline Model
# ==========================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

baseline_model = DummyClassifier(strategy="most_frequent")

baseline_model.fit(X_train_processed, y_train)

y_pred_baseline = baseline_model.predict(X_test_processed)

baseline_accuracy = accuracy_score(
    y_test,
    y_pred_baseline
)

print("\n========== BASELINE MODEL ==========")
print("Accuracy:", baseline_accuracy)

# ==========================================
# 9. Logistic Regression
# ==========================================

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_processed, y_train)

y_pred_logistic = logistic_model.predict(X_test_processed)

print("\n========== LOGISTIC REGRESSION ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_logistic))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))

# ==========================================
# 10. Balanced Logistic Regression
# ==========================================

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

balanced_model.fit(X_train_processed, y_train)

y_pred_balanced = balanced_model.predict(X_test_processed)

print("\n========== BALANCED LOGISTIC REGRESSION ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_balanced))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))

# ==========================================
# 11. Decision Tree (Limited Depth)
# ==========================================

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

tree_model.fit(X_train_processed, y_train)

y_pred_tree = tree_model.predict(X_test_processed)

print("\n========== DECISION TREE ==========")

print("Accuracy:",
      accuracy_score(y_test, y_pred_tree))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

# Overfitting Check

y_train_pred_tree = tree_model.predict(X_train_processed)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred_tree
)

test_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

print("\n========== OVERFITTING CHECK ==========")

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("Tree Depth:", tree_model.get_depth())
print("Number of Leaves:", tree_model.get_n_leaves())

# Feature Importance

importances = tree_model.feature_importances_

feature_names = preprocessor.get_feature_names_out()

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance_df.head(10).to_string(index=False))

# Cross-Validation

# Comparing Models with Cross-Validation

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    )
}

print("\n========== MODEL COMPARISON ==========")

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Detailed Model Evaluation with Cross-Validation

from sklearn.model_selection import cross_validate

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

print("\n========== DETAILED MODEL COMPARISON ==========")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    print(f"\n{name}")

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric.capitalize()}: "
            f"{scores.mean():.4f}"
        )

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(f"\n{name}")
    print("Fold Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    print("Standard Deviation:", scores.std())

from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

scoring_classes = {
    "pass_recall": "recall",
    "pass_precision": "precision",
    "fail_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
    "fail_precision": make_scorer(precision_score, pos_label=0, zero_division=0),
    "fail_f1": make_scorer(f1_score, pos_label=0, zero_division=0)
}

print("\n========== PASS / FAIL PERFORMANCE ==========")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring_classes
    )

    print(f"\n{name}")

    for metric in scoring_classes:
        print(
            f"{metric}: "
            f"{results['test_' + metric].mean():.4f}"
        )    
from sklearn.model_selection import cross_validate

scoring = {
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}

print("\n========== MACRO METRICS COMPARISON ==========")

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    print(f"\n{name}")

    for metric in scoring:
        scores = results[f"test_{metric}"]

        print(
            f"{metric}: {scores.mean():.4f}"
        )
# ==========================================
# 12. Visualize Decision Tree (Simplified)
# ==========================================

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 7))

plot_tree(
    tree_model,
    max_depth=2,
    filled=True,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=["Fail", "Pass"],
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - First 2 Levels")

plt.tight_layout()
plt.show()

plt.close()

from sklearn.model_selection import cross_val_score

print("\n========== HYPERPARAMETER TUNING ==========")

depth_values = [2, 3, 4, 5, 6, 8]

for depth in depth_values:

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    print(
        f"Depth: {depth} | "
        f"Mean Accuracy: {scores.mean():.4f} | "
        f"Std: {scores.std():.4f}"
    )
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    "model__max_depth": [2, 3, 4, 5, 6, 8],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\n========== GRID SEARCH RESULTS ==========")

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test)

# Evaluate final model
print("\n========== FINAL MODEL EVALUATION ==========")

print("Test Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Balanced Logistic Regression Final Evaluation

balanced_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

balanced_model.fit(X_train, y_train)

balanced_pred = balanced_model.predict(X_test)

print("\n========== BALANCED LOGISTIC TEST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, balanced_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, balanced_pred))

print("\nClassification Report:")
print(classification_report(y_test, balanced_pred))

print("\n========== ERROR ANALYSIS: DECISION TREE ==========")

# پیش‌بینی مدل نهایی روی داده‌های تست
tree_predictions = best_model.predict(X_test)

# ساخت جدول برای مقایسه پیش‌بینی با واقعیت
errors = X_test.copy()

errors["Actual"] = y_test
errors["Predicted"] = tree_predictions

# جدا کردن پیش‌بینی‌های اشتباه
wrong_predictions = errors[
    errors["Actual"] != errors["Predicted"]
].copy()

# تبدیل اعداد به برچسب‌های قابل فهم
wrong_predictions["Actual"] = wrong_predictions["Actual"].map({
    0: "Fail",
    1: "Pass"
})

wrong_predictions["Predicted"] = wrong_predictions["Predicted"].map({
    0: "Fail",
    1: "Pass"
})

print("Total Test Samples:", len(X_test))
print("Total Wrong Predictions:", len(wrong_predictions))

print("\nWrong Predictions Details:")

print(
    wrong_predictions[
        ["Actual", "Predicted", "failures", "absences", "studytime"]
    ].to_string()
)

print("\nError Types:")

print(
    wrong_predictions.groupby(
        ["Actual", "Predicted"]
    ).size()
)

print("\n========== ERROR ANALYSIS: BALANCED LOGISTIC ==========")

# ساخت جدول پیش‌بینی‌ها
balanced_errors = X_test.copy()

balanced_errors["Actual"] = y_test
balanced_errors["Predicted"] = balanced_pred

# جدا کردن پیش‌بینی‌های اشتباه
wrong_balanced = balanced_errors[
    balanced_errors["Actual"] != balanced_errors["Predicted"]
].copy()

print("Total Test Samples:", len(X_test))
print("Total Wrong Predictions:", len(wrong_balanced))

# نمایش جزئیات خطاها
print("\nWrong Predictions Details:")

print(
    wrong_balanced[
        ["Actual", "Predicted", "failures", "absences", "studytime"]
    ].to_string()
)

# شمارش انواع خطا
print("\nError Types:")

print(
    wrong_balanced.groupby(
        ["Actual", "Predicted"]
    ).size()
)
print("\n========== FINAL MODEL COMPARISON ==========")

final_results = {
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Tuned Decision Tree"
    ],

    "Accuracy": [
        0.6582,
        0.5949,
        0.6582
    ],

    "Fail Recall": [
        0.3462,
        0.4615,
        0.3462
    ],

    "Fail Precision": [
        0.4737,
        0.4000,
        0.4737
    ],

    "Fail F1": [
        0.4000,
        0.4300,
        0.4000
    ],

    "Pass Recall": [
        0.8113,
        0.6604,
        0.8113
    ]
}

final_df = pd.DataFrame(final_results)

print(final_df.to_string(index=False))
