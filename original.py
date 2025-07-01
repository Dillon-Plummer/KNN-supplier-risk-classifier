import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

SC_increase_percentage = 4
QA_increase_percentage = 1

### DF OPTION 1 ###
df = pd.DataFrame({
    "% of NCMRs per Total Lots": [round(27 * QA_increase_percentage)] * 350,
    "Shipment Inaccuracy": [round(28 * SC_increase_percentage)] * 350,
    "Failed OTD": [round(24 * SC_increase_percentage)] * 350,
    "Audit Findings": [round(24 * QA_increase_percentage)] * 350,
    "Financial Obstacles": [round(24 * SC_increase_percentage)] * 350,
    "Response Delay (Docs)": [round(24 * SC_increase_percentage)] * 350,
    "Capacity Limit": [round(24 * SC_increase_percentage)] * 350,
    "Lack of Documentation (CoC, etc.)": [round(24 * QA_increase_percentage)] * 350,
    "Unjustified Price Increase": [round(24 * SC_increase_percentage)] * 350,
    "No Cost Reduction Participation": [round(24 * SC_increase_percentage)] * 350,
})

df["Risk Classification"] = 2

# Generate random values for columns within 1 standard deviation
for col in df.columns:
    base_value = df[col].mean()
    std = base_value * 0.1
    df[col] = [round(random.normalvariate(base_value, std)) for _ in range(350)]

# # Adjust thresholds to create more balanced risk distribution
# # SC = 3; QA = 1
# # df option 1
# high_threshold_NCMR = 40
# high_threshold_shipment = 85
# high_threshold_OTD = 85
# high_threshold_audit = 50
# high_threshold_finance = 85
# high_threshold_response_delay = 115
# high_threshold_capacity = 105
# high_threshold_CoCs = 45
# high_threshold_price_increase = 110
# high_threshold_cost_reduction = 100
################################################
# SC = 4; QA = 1
high_threshold_NCMR = 40
high_threshold_shipment = 85
high_threshold_OTD = 85
high_threshold_audit = 50
high_threshold_finance = 85
high_threshold_response_delay = 105
high_threshold_capacity = 105
high_threshold_CoCs = 45
high_threshold_price_increase = 100
high_threshold_cost_reduction = 100

# Assign Risk Based on Conditions (revised to consider all variables)
for index, row in df.iterrows():
    high_count = 0

    # Check each column and increment counter
    for col in df.columns:
        if row[col] >= high_threshold_NCMR and col == "% of NCMRs per Total Lots":
            high_count += 1
        elif row[col] >= high_threshold_shipment and col == "Shipment Inaccuracy":
            high_count += 1
        elif row[col] >= high_threshold_OTD and col == "Failed OTD":
            high_count += 1
        elif row[col] >= high_threshold_audit and col == "Audit Findings":
            high_count += 1
        elif row[col] >= high_threshold_finance and col == "Financial Obstacles":
            high_count += 1
        elif row[col] >= high_threshold_response_delay and col == "Response Delay (Docs)":
            high_count += 1
        elif row[col] >= high_threshold_capacity and col == "Capacity Limit":
            high_count += 1
        elif row[col] >= high_threshold_CoCs and col == "Lack of Documentation (CoC, etc.)":
            high_count += 1
        elif row[col] >= high_threshold_price_increase and col == "Unjustified Price Increase":
            high_count += 1
        elif row[col] >= high_threshold_cost_reduction and col == "No Cost Reduction Participation":
            high_count += 1

    # Assign risk based on counter value
    if high_count == 3:
        df.loc[index, "Risk Classification"] = 1  # All 3 exceed
    elif high_count == 2:
        df.loc[index, "Risk Classification"] = 2  # 2 out of 3 exceed
    elif high_count == 1:
        df.loc[index, "Risk Classification"] = 2  # 1 out of 3 exceeds
    else:
        df.loc[index, "Risk Classification"] = 3  # None exceed

# Get maximum value across all columns for Y axis scaling
max_value = df.max().max()

############################################################################

# set variables
X = df
y = df["Risk Classification"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features using StandardScaler (on training set only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model with prediction and evaluation (using test set labels)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
risk_mapping = {3: "Low Risk", 2: "Medium Risk", 1: "High Risk"}
X["Risk Category"] = X["Risk Classification"].map(risk_mapping)

# Create histograms for each column with shared Y axis limits
plt.figure(figsize=(16, 9))
for i, col in enumerate(X.columns):
    ax = plt.subplot(5, 3, i+1)
    sns.histplot(x=col, data=X, ax=ax, bins=len(df.columns))
    ax.set_ylim(0, max_value)
    ax.set_xlim(0, max_value)
    ax.set_title(col)
plt.tight_layout()
plt.show()

sns.histplot(data=df, x=df['Risk Classification'])
plt.show()

# KNN Pairplot
color_pallete = ['green', 'red', 'orange']
sns.pairplot(X, hue="Risk Category", palette=color_pallete, plot_kws={"alpha": 0.65})
plt.show()

####################################################################################
### TESTING ###
###############

accuracy = accuracy_score(y_test, y_pred)

# Precision, Recall, F1-score (macro and micro)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the classification report to a dataframe
evaluation_df = pd.DataFrame(report).transpose()

evaluation_df
