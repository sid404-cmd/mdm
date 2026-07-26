# ==========================================
# Import Libraries
# ==========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ==========================================
# Load Dataset
# ==========================================
data = pd.read_csv("heart.csv")

# ==========================================
# Separate Features and Target
# ==========================================
X = data.drop("target", axis=1)
y = data["target"]

# ==========================================
# Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Feature Scaling
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# Train Logistic Regression Model
# ==========================================
model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

model.fit(X_train, y_train)

# ==========================================
# Predictions
# ==========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# Evaluation Metrics
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("=" * 40)
print("Baseline Logistic Regression")
print("=" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

# ==========================================
# Feature Coefficients
# ==========================================
coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coef["Absolute Coefficient"] = coef["Coefficient"].abs()

coef = coef.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\nFeature Importance (Logistic Regression Coefficients)\n")
print(coef)