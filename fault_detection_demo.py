"""
Demo code for:
Machine Learning Based Fault Detection in Electrical Equipment

This is a sample implementation inspired by the paper methodology.
It creates a synthetic electrical fault dataset, preprocesses the data,
trains multiple machine learning models, and evaluates the best model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_fault_data(n_samples=3000):
    """
    Generates synthetic three-phase electrical equipment data.

    Features:
    Ia, Ib, Ic: phase currents
    Va, Vb, Vc: phase voltages
    P: active power
    R: equivalent resistance

    Target:
    0 = No Fault
    1 = Line-to-Ground Fault
    2 = Line-to-Line Fault
    3 = Three-Phase Fault
    """

    fault_type = np.random.choice(
        [0, 1, 2, 3],
        size=n_samples,
        p=[0.35, 0.25, 0.25, 0.15]
    )

    data = []

    for fault in fault_type:
        if fault == 0:
            ia = np.random.normal(100, 8)
            ib = np.random.normal(100, 8)
            ic = np.random.normal(100, 8)
            va = np.random.normal(230, 5)
            vb = np.random.normal(230, 5)
            vc = np.random.normal(230, 5)

        elif fault == 1:
            ia = np.random.normal(180, 20)
            ib = np.random.normal(105, 10)
            ic = np.random.normal(100, 10)
            va = np.random.normal(140, 15)
            vb = np.random.normal(225, 8)
            vc = np.random.normal(230, 8)

        elif fault == 2:
            ia = np.random.normal(170, 18)
            ib = np.random.normal(165, 18)
            ic = np.random.normal(95, 10)
            va = np.random.normal(160, 15)
            vb = np.random.normal(155, 15)
            vc = np.random.normal(230, 8)

        else:
            ia = np.random.normal(210, 25)
            ib = np.random.normal(205, 25)
            ic = np.random.normal(208, 25)
            va = np.random.normal(120, 20)
            vb = np.random.normal(125, 20)
            vc = np.random.normal(118, 20)

        power = (ia * va + ib * vb + ic * vc) / 1000
        resistance = np.mean([va / max(ia, 1), vb / max(ib, 1), vc / max(ic, 1)])

        data.append([ia, ib, ic, va, vb, vc, power, resistance, fault])

    columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "Power_kW", "Resistance", "Fault_Type"]
    return pd.DataFrame(data, columns=columns)


def remove_outliers_iqr(df, feature_columns):
    """
    Removes outliers using the interquartile range method.
    """
    clean_df = df.copy()

    for col in feature_columns:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        clean_df = clean_df[(clean_df[col] >= lower) & (clean_df[col] <= upper)]

    return clean_df


def plot_confusion_matrix(cm, labels, output_file="confusion_matrix.png"):
    """
    Saves a confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    print("Generating synthetic electrical fault dataset...")
    df = generate_synthetic_fault_data(n_samples=3000)

    feature_columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "Power_kW", "Resistance"]
    target_column = "Fault_Type"

    print("\nOriginal dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df[target_column].value_counts().sort_index())

    df = df.dropna()
    df = remove_outliers_iqr(df, feature_columns)

    print("\nCleaned dataset shape:", df.shape)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE
        )
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            "accuracy": accuracy,
            "pipeline": pipeline,
            "predictions": predictions
        }

        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

    best_model_name = max(results, key=lambda name: results[name]["accuracy"])
    best_predictions = results[best_model_name]["predictions"]

    print("\nBest model:", best_model_name)
    print("Best accuracy:", round(results[best_model_name]["accuracy"], 4))

    labels = ["No Fault", "Line-Ground", "Line-Line", "Three-Phase"]
    cm = confusion_matrix(y_test, best_predictions)
    plot_confusion_matrix(cm, labels)

    df.to_csv("synthetic_fault_dataset.csv", index=False)

    print("\nSaved files:")
    print("synthetic_fault_dataset.csv")
    print("confusion_matrix.png")


if __name__ == "__main__":
    main()
