import pandas as pd
import numpy as np
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

TARGET_COLUMN = "Risk Classification"

def generate_dataset(n_samples=350, overlap_multiplier=0.40):
    """
    Create a dataset with distinct families/clusters for each risk class.
    """
    data_frames = []
    risk_profiles = {
        "High": (3, 1.6, n_samples // 3),
        "Medium": (2, 1.0, n_samples // 3),
        "Low": (1, 0.4, 0)
    }
    n_low = n_samples - (risk_profiles["High"][2] + risk_profiles["Medium"][2])
    risk_profiles["Low"] = (1, 0.4, n_low)

    base_value = 50
    features = ["% of NCMRs per Total Lots", "Shipment Inaccuracy", "Failed OTD", "Audit Findings", "Financial Obstacles"]

    for profile_name, (label, multiplier, count) in risk_profiles.items():
        if count == 0: continue
        class_df = pd.DataFrame()
        for feature in features:
            center = base_value * multiplier
            std_dev = center * overlap_multiplier
            class_df[feature] = np.random.normal(loc=center, scale=std_dev, size=count)
        class_df[TARGET_COLUMN] = label
        data_frames.append(class_df)

    final_df = pd.concat(data_frames, ignore_index=True)
    final_df[features] = final_df[features].clip(lower=0).round().astype(int)
    return final_df.sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess_data(df):
    """
    Cleans and preprocesses the input dataframe.
    """
    # Handle duplicated column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    # --- FIX: Ensure the target column is clean and numeric from the start ---
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Error: Target column '{TARGET_COLUMN}' not found.")
    
    # Coerce non-numeric values in the target column to NaN
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    
    # Now, drop any rows that have NaN values (including newly coerced ones)
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    
    # Ensure the clean target column is an integer
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    # --- END FIX ---

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    categorical_features = features.select_dtypes(include=['object', 'category']).columns
    if not categorical_features.empty:
        features_encoded = pd.get_dummies(features, columns=categorical_features, drop_first=True)
    else:
        features_encoded = features

    processed_df = pd.concat([features_encoded, target], axis=1)
    
    return processed_df, dropped_rows, list(categorical_features)


def train_knn(df, n_neighbors=3):
    """
    Train a KNN model and return evaluation metrics.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return metrics, pd.DataFrame(report).transpose(), cm_df, knn, scaler
