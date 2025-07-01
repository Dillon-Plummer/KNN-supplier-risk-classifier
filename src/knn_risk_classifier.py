import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_dataset(sc_increase_percentage=4, qa_increase_percentage=1, n_samples=350):
    """Create a randomized dataset for supplier risk classification."""
    df = pd.DataFrame({
        "% of NCMRs per Total Lots": [round(27 * qa_increase_percentage)] * n_samples,
        "Shipment Inaccuracy": [round(28 * sc_increase_percentage)] * n_samples,
        "Failed OTD": [round(24 * sc_increase_percentage)] * n_samples,
        "Audit Findings": [round(24 * qa_increase_percentage)] * n_samples,
        "Financial Obstacles": [round(24 * sc_increase_percentage)] * n_samples,
        "Response Delay (Docs)": [round(24 * sc_increase_percentage)] * n_samples,
        "Capacity Limit": [round(24 * sc_increase_percentage)] * n_samples,
        "Lack of Documentation (CoC, etc.)": [round(24 * qa_increase_percentage)] * n_samples,
        "Unjustified Price Increase": [round(24 * sc_increase_percentage)] * n_samples,
        "No Cost Reduction Participation": [round(24 * sc_increase_percentage)] * n_samples,
    })

    df["Risk Classification"] = 2

    for col in df.columns:
        base_value = df[col].mean()
        std = base_value * 0.1
        df[col] = [round(random.normalvariate(base_value, std)) for _ in range(n_samples)]

    return df


def assign_risk(df):
    """Assign risk levels based on threshold counts."""
    thresholds = {
        "% of NCMRs per Total Lots": 40,
        "Shipment Inaccuracy": 85,
        "Failed OTD": 85,
        "Audit Findings": 50,
        "Financial Obstacles": 85,
        "Response Delay (Docs)": 105,
        "Capacity Limit": 105,
        "Lack of Documentation (CoC, etc.)": 45,
        "Unjustified Price Increase": 100,
        "No Cost Reduction Participation": 100,
    }

    for index, row in df.iterrows():
        high_count = sum(row[col] >= thresholds[col] for col in thresholds)
        if high_count >= 3:
            df.at[index, "Risk Classification"] = 1
        elif high_count >= 1:
            df.at[index, "Risk Classification"] = 2
        else:
            df.at[index, "Risk Classification"] = 3
    return df


def train_knn(df, n_neighbors=3):
    """Train a KNN model and return evaluation metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with features and ``Risk Classification`` column.
    n_neighbors : int, optional
        Number of neighbours for ``KNeighborsClassifier``.

    Returns
    -------
    tuple
        Tuple containing metrics dictionary, classification report
        ``pandas.DataFrame`` and a confusion matrix ``pandas.DataFrame``.
    """
    X = df.drop(columns=["Risk Classification"])
    y = df["Risk Classification"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    report = classification_report(y_test, y_pred, output_dict=True)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    return metrics, pd.DataFrame(report).transpose(), cm_df


if __name__ == "__main__":
    df = generate_dataset()
    df = assign_risk(df)
    metrics, evaluation_df, cm_df = train_knn(df)
    print(metrics)
    print(evaluation_df)
    print(cm_df)
