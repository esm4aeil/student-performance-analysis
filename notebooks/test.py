"""Student performance analysis: exploratory analysis and pass/fail models.

Run from the project root:
    python notebooks/test.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "student-mat.csv"
FIGURES_DIR = PROJECT_DIR / "figures"

RANDOM_STATE = 42
PASS_MARK = 10
CLASS_NAMES = ["Fail", "Pass"]  # encoded as 0 and 1
TREE_DEPTHS = [2, 3, 4, 5, 6, 8]
# Scorer settings that treat "Fail" as the positive class
FAIL_CLASS = {"pos_label": 0, "zero_division": 0}

MAIN_COLOR = "#257de0"
METRIC_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600"]


def print_header(title):
    print(f"\n========== {title} ==========")


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=150)


# ----- Data exploration -----
def describe_data(df):
    print("First 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

    print("\nFinal Grade Statistics:")
    print(df["G3"].describe())

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    passed = (df["G3"] >= PASS_MARK).sum()
    print("\nPass/Fail Statistics:")
    print("Passed:", passed)
    print("Failed:", len(df) - passed)


def plot_mean_grade(summary, tick_labels, xlabel, title, filename):
    """Bar chart of the mean final grade per group, with group sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tick_labels, summary["mean"], color=MAIN_COLOR, width=0.6)
    ax.bar_label(
        bars,
        labels=[
            f"{row['mean']:.2f}\n(n={row['count']:.0f})"
            for _, row in summary.iterrows()
        ],
        padding=3,
    )
    ax.set(xlabel=xlabel, ylabel="Mean final grade (G3)", title=title)
    ax.set_ylim(0, summary["mean"].max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, filename)


def plot_absences(df, correlation):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["absences"], df["G3"], color=MAIN_COLOR, alpha=0.5)
    ax.set(
        xlabel="Absences",
        ylabel="Final grade (G3)",
        title=f"Absences vs. Final Grade (r = {correlation:.3f})",
    )
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "absences_vs_grade.png")


def explore_relationships(df):
    study_summary = df.groupby("studytime")["G3"].agg(["count", "mean"])
    print("\nAverage Grade by Study Time:")
    print(study_summary["mean"].rename("G3"))

    correlation = df["absences"].corr(df["G3"])
    print("\nCorrelation between Absences and G3:")
    print(correlation)

    failure_summary = df.groupby("failures")["G3"].agg(
        ["count", "mean", "median"]
    )
    print("\nFailure Analysis:")
    print(failure_summary)

    study_labels = {1: "< 2 h", 2: "2-5 h", 3: "5-10 h", 4: "> 10 h"}
    plot_mean_grade(
        study_summary,
        [study_labels[level] for level in study_summary.index],
        "Weekly study time",
        "Final Grade by Study Time",
        "grade_by_study_time.png",
    )
    plot_absences(df, correlation)
    plot_mean_grade(
        failure_summary,
        failure_summary.index.astype(str),
        "Previous class failures",
        "Final Grade by Previous Failures",
        "grade_by_failures.png",
    )


# ----- Preprocessing -----
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include="number").columns
    categorical_features = X.select_dtypes(exclude="number").columns

    print("\nNumerical Features:", len(numeric_features))
    print("Categorical Features:", len(categorical_features))

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def build_pipeline(preprocessor, model):
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ----- Models and evaluation -----
def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Balanced Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=RANDOM_STATE
        ),
    }


def print_evaluation(title, y_true, y_pred, accuracy_label="Accuracy"):
    print_header(title)
    print(f"{accuracy_label}:", accuracy_score(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def report_feature_importance(tree, feature_names):
    importance = pd.DataFrame(
        {"Feature": feature_names, "Importance": tree.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    top_features = importance.head(10)

    print_header("FEATURE IMPORTANCE")
    print(top_features.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        top_features["Feature"], top_features["Importance"], color=MAIN_COLOR
    )
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.invert_yaxis()
    ax.set(
        xlabel="Importance",
        title=f"Top 10 Features - Decision Tree (depth {tree.get_depth()})",
    )
    ax.set_xlim(0, top_features["Importance"].max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "feature_importance.png")


def plot_decision_tree(tree, feature_names):
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(
        tree,
        max_depth=2,
        filled=True,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree - First 2 Levels")
    save_figure(fig, "decision_tree.png")


def cross_validate_models(models, preprocessor, X, y, cv):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "fail_recall": make_scorer(recall_score, **FAIL_CLASS),
        "fail_precision": make_scorer(precision_score, **FAIL_CLASS),
        "fail_f1": make_scorer(f1_score, **FAIL_CLASS),
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    results = {
        name: cross_validate(
            build_pipeline(preprocessor, model),
            X,
            y,
            cv=cv,
            scoring=scoring,
        )
        for name, model in models.items()
    }

    print_header("DETAILED MODEL COMPARISON")
    for name, scores in results.items():
        print(f"\n{name}")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            mean_score = scores[f"test_{metric}"].mean()
            print(f"{metric.capitalize()}: {mean_score:.4f}")

    for name, scores in results.items():
        fold_scores = scores["test_accuracy"]
        print(f"\n{name}")
        print("Fold Scores:", fold_scores)
        print("Mean Accuracy:", fold_scores.mean())
        print("Standard Deviation:", fold_scores.std())

    # Printed label -> scorer name. "Pass" is the positive class.
    class_metrics = {
        "pass_recall": "recall",
        "pass_precision": "precision",
        "fail_recall": "fail_recall",
        "fail_precision": "fail_precision",
        "fail_f1": "fail_f1",
    }
    print_header("PASS / FAIL PERFORMANCE")
    for name, scores in results.items():
        print(f"\n{name}")
        for label, metric in class_metrics.items():
            print(f"{label}: {scores[f'test_{metric}'].mean():.4f}")

    print_header("MACRO METRICS COMPARISON")
    for name, scores in results.items():
        print(f"\n{name}")
        for metric in ["precision_macro", "recall_macro", "f1_macro"]:
            print(f"{metric}: {scores[f'test_{metric}'].mean():.4f}")


def compare_tree_depths(preprocessor, X, y, cv):
    print_header("HYPERPARAMETER TUNING")
    for depth in TREE_DEPTHS:
        tree = DecisionTreeClassifier(
            max_depth=depth, random_state=RANDOM_STATE
        )
        scores = cross_val_score(
            build_pipeline(preprocessor, tree),
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )
        print(
            f"Depth: {depth} | "
            f"Mean Accuracy: {scores.mean():.4f} | "
            f"Std: {scores.std():.4f}"
        )


def tune_decision_tree(preprocessor, X, y, cv):
    pipeline = build_pipeline(
        preprocessor, DecisionTreeClassifier(random_state=RANDOM_STATE)
    )
    param_grid = {
        "model__max_depth": TREE_DEPTHS,
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )
    grid_search.fit(X, y)

    print_header("GRID SEARCH RESULTS")
    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Accuracy:", grid_search.best_score_)

    return grid_search.best_estimator_


def analyze_errors(title, X_test, y_test, y_pred, use_class_names=False):
    results = X_test.assign(Actual=y_test, Predicted=y_pred)
    errors = results[results["Actual"] != results["Predicted"]]

    if use_class_names:
        names = dict(enumerate(CLASS_NAMES))
        errors = errors.assign(
            Actual=errors["Actual"].map(names),
            Predicted=errors["Predicted"].map(names),
        )

    print_header(f"ERROR ANALYSIS: {title}")
    print("Total Test Samples:", len(X_test))
    print("Total Wrong Predictions:", len(errors))

    print("\nWrong Predictions Details:")
    columns = ["Actual", "Predicted", "failures", "absences", "studytime"]
    print(errors[columns].to_string())

    print("\nError Types:")
    print(errors.groupby(["Actual", "Predicted"]).size())


def plot_confusion_matrices(y_test, predictions):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, y_pred) in zip(axes.flat, predictions.items()):
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=CLASS_NAMES,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        # Use one color scale so the four panels are comparable
        display.im_.set_clim(0, len(y_test))
        counts = display.confusion_matrix.flat
        for count, text in zip(counts, display.text_.flat):
            text.set_color("white" if count > len(y_test) / 2 else "black")
        ax.set_title(name)
    fig.suptitle("Confusion Matrices (Test Set)")
    save_figure(fig, "confusion_matrices.png")


def summarize_test_results(y_test, predictions):
    rows = []
    for name, y_pred in predictions.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Fail Recall": recall_score(y_test, y_pred, **FAIL_CLASS),
                "Fail Precision": precision_score(
                    y_test, y_pred, **FAIL_CLASS
                ),
                "Fail F1": f1_score(y_test, y_pred, **FAIL_CLASS),
                "Pass Recall": recall_score(y_test, y_pred),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def plot_model_comparison(results):
    metrics = ["Accuracy", "Fail Recall", "Fail Precision", "Fail F1"]
    fig, ax = plt.subplots(figsize=(12, 6))
    results[metrics].plot.bar(
        ax=ax, color=METRIC_COLORS, width=0.8, edgecolor="white", rot=0
    )
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    # Wrap long model names so the tick labels do not overlap
    ax.set_xticklabels([name.replace(" ", "\n", 1) for name in results.index])
    ax.set(xlabel="", ylabel="Score", title="Model Comparison (Test Set)")
    ax.set_ylim(0, 1)
    ax.legend(ncols=len(metrics), loc="upper center", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, "model_comparison.png")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")
    describe_data(df)
    explore_relationships(df)

    # Target: 1 = pass, 0 = fail
    y = (df["G3"] >= PASS_MARK).astype(int)
    # Exclude all grade columns to avoid leakage
    X = df.drop(columns=["G1", "G2", "G3"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
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

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing Results:")
    print("X_train_processed:", X_train_processed.shape)
    print("X_test_processed:", X_test_processed.shape)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train_processed, y_train)
    y_pred_baseline = baseline.predict(X_test_processed)

    print_header("BASELINE MODEL")
    print("Accuracy:", accuracy_score(y_test, y_pred_baseline))

    models = build_models()
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        predictions[name] = model.predict(X_test_processed)
        print_evaluation(name.upper(), y_test, predictions[name])

    tree = models["Decision Tree"]
    train_accuracy = accuracy_score(y_train, tree.predict(X_train_processed))

    print_header("OVERFITTING CHECK")
    print("Training Accuracy:", train_accuracy)
    print(
        "Testing Accuracy:",
        accuracy_score(y_test, predictions["Decision Tree"]),
    )
    print("Tree Depth:", tree.get_depth())
    print("Number of Leaves:", tree.get_n_leaves())

    feature_names = preprocessor.get_feature_names_out()
    report_feature_importance(tree, feature_names)
    plot_decision_tree(tree, feature_names)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cross_validate_models(models, preprocessor, X_train, y_train, cv)
    compare_tree_depths(preprocessor, X_train, y_train, cv)
    tuned_tree = tune_decision_tree(preprocessor, X_train, y_train, cv)

    y_pred_tuned = tuned_tree.predict(X_test)
    print_evaluation(
        "FINAL MODEL EVALUATION",
        y_test,
        y_pred_tuned,
        accuracy_label="Test Accuracy",
    )

    y_pred_balanced = predictions["Balanced Logistic Regression"]
    analyze_errors(
        "DECISION TREE", X_test, y_test, y_pred_tuned, use_class_names=True
    )
    analyze_errors("BALANCED LOGISTIC", X_test, y_test, y_pred_balanced)

    final_predictions = {
        "Dummy Baseline": y_pred_baseline,
        "Logistic Regression": predictions["Logistic Regression"],
        "Balanced Logistic Regression": y_pred_balanced,
        "Tuned Decision Tree": y_pred_tuned,
    }
    test_results = summarize_test_results(y_test, final_predictions)

    print_header("FINAL MODEL COMPARISON")
    print(
        test_results.drop(index="Dummy Baseline")
        .round(4)
        .reset_index()
        .to_string(index=False)
    )

    plot_confusion_matrices(y_test, final_predictions)
    plot_model_comparison(test_results)

    print(f"\nFigures saved to: {FIGURES_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
