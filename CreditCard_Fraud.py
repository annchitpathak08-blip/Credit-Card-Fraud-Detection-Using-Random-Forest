# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load Dataset

df = pd.read_csv("creditcard.csv")

# Data Acquisition

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Information")
df.info()     # Dataset contains information of about quarter a million people 

print("\nStatistical Summary")
print(df.describe())

# Data Manipulation & Analysis (DMA)

print("\nMissing Values")
print(df.isnull().sum())  # no missing values found

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nClass Distribution")
print(df["Class"].value_counts())

print("\nClass Percentage")
print(df["Class"].value_counts(normalize=True) * 100)

# Remove Duplicate Rows

df = df.drop_duplicates()
# no need for scaling or encoding since random forest works directly w/ numeric values
# Split Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]
# No need to scale or encode a random forest therefore skip that

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest Model

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced", # to make sure the model doesnt cheat its way out
    random_state=42
)

# Train Model

model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix")
plt.show()

# ROC Curve

RocCurveDisplay.from_predictions(y_test, y_prob)

plt.title("ROC Curve")
plt.show()

# Feature Importance
# Calculates and sorts the features based on information
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind="bar")

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.tight_layout()
plt.show()

# Accuracy  : 0.9995
# Precision : 0.9452
# Recall    : 0.7263
# F1 Score  : 0.8214
# ROC-AUC   : 0.9391         For reference