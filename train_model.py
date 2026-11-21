
# ==========================================
# MACHINE PROBLEM 3
# Student Performance Prediction
# Model Training
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. LOAD DATASET
df = pd.read_csv("data/augmented_balanced_dataset_m.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# 2. REMOVE EXACT DUPLICATES
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 3. DEFINE FEATURES AND TARGET
features = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In"
]

X = df[features]
y = df["rating"]

# 4. SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. CREATE MACHINE LEARNING PIPELINE
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 6. TRAIN MODEL
model.fit(X_train, y_train)

print("\nModel training completed!")

# 7. MAKE PREDICTIONS
y_pred = model.predict(X_test)

# 8. MODEL EVALUATION
print("\n========== MODEL EVALUATION ==========")

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. FEATURE IMPORTANCE
classifier = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": classifier.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# 10. SAVE TRAINED MODEL
joblib.dump(model, "student_model.pkl")

print("\nModel saved as student_model.pkl")