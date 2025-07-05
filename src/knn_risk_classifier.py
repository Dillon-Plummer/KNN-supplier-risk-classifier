import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



def generate_dataset(sc_increase_percentage=4, qa_increase_percentage=1, n_samples=350):
    """
    Create a dataset with distinct families/clusters for each risk class.
    This function directly generates the classes, so the `assign_risk` function is no longer needed.
    """
    data_frames = []

    # Define the characteristics for each risk profile
    # Format: { "Display Name": (class_label, value_multiplier, num_samples) }
    risk_profiles = {
        "High": (1, 1.6, n_samples // 3),
        "Medium": (2, 1.0, n_samples // 3),
    }
    # Calculate remaining samples for the Low risk group
    n_low = n_samples - sum(p[2] for p in risk_profiles.values())
    risk_profiles["Low"] = (3, 0.4, n_low)

    # Define base values using the sidebar sliders
    base_sc_value = 25 * sc_increase_percentage
    base_qa_value = 25 * qa_increase_percentage

    # Group features for easier manipulation
    sc_features = ["Shipment Inaccuracy", "Failed OTD", "Financial Obstacles", "Response Delay (Docs)", "Capacity Limit", "Unjustified Price Increase", "No Cost Reduction Participation"]
    qa_features = ["% of NCMRs per Total Lots", "Audit Findings", "Lack of Documentation (CoC, etc.)"]

    for profile_name, (label, multiplier, count) in risk_profiles.items():
        if count == 0:
            continue
            
        class_df = pd.DataFrame()
        # Generate skewed data for each feature group
        for feature in sc_features:
            center = base_sc_value * multiplier
            std_dev = center * 0.10 # Lower standard deviation for tighter clusters
            class_df[feature] = np.random.normal(loc=center, scale=std_dev, size=count)
        
        for feature in qa_features:
            center = base_qa_value * multiplier
            std_dev = center * 0.10
            class_df[feature] = np.random.normal(loc=center, scale=std_dev, size=count)
        
        class_df["Risk Classification"] = label
        data_frames.append(class_df)

    # Combine the data for each class into a single DataFrame
    final_df = pd.concat(data_frames, ignore_index=True)
    
    # Clean up the data: ensure values are non-negative integers
    feature_cols = sc_features + qa_features
    final_df[feature_cols] = final_df[feature_cols].clip(lower=0).round().astype(int)

    # Shuffle the dataset to mix the classes together
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)



# def assign_risk(df):
#     """Assign risk levels based on threshold counts."""
#     thresholds = {
#         "% of NCMRs per Total Lots": 40,
#         "Shipment Inaccuracy": 85,
#         "Failed OTD": 85,
#         "Audit Findings": 50,
#         "Financial Obstacles": 85,
#         "Response Delay (Docs)": 105,
#         "Capacity Limit": 105,
#         "Lack of Documentation (CoC, etc.)": 45,
#         "Unjustified Price Increase": 100,
#         "No Cost Reduction Participation": 100,
#     }

#     for index, row in df.iterrows():
#         high_count = sum(row[col] >= thresholds[col] for col in thresholds)
#         if high_count >= 3:
#             df.at[index, "Risk Classification"] = 1
#         elif high_count >= 1:
#             df.at[index, "Risk Classification"] = 2
#         else:
#             df.at[index, "Risk Classification"] = 3
#     return df


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

    return metrics, pd.DataFrame(report).transpose(), cm_df, knn, scaler


if __name__ == "__main__":
    df = generate_dataset()
    df = assign_risk(df)
    metrics, evaluation_df, cm_df, model, scaler = train_knn(df)
    print(metrics)
    print(evaluation_df)
    print(cm_df)
