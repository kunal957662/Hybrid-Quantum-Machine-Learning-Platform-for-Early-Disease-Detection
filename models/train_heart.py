import os
import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. LOAD UCI HEART DISEASE DATASET
# ============================================================

print("\nLoading Heart Disease dataset...")

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()


# ============================================================
# 2. CLEAN TARGET
# ============================================================

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

y = pd.to_numeric(y, errors="coerce")

# UCI Heart Disease:
# 0 = no disease
# 1,2,3,4 = disease
y = (y > 0).astype(int)


# ============================================================
# 3. CLEAN FEATURES
# ============================================================

print("\nOriginal shape:")
print(X.shape)

# Convert everything to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# Replace missing values with median
X = X.fillna(X.median())


print("\nMissing values after cleaning:")
print(X.isnull().sum())

print("\nFeatures:")
print(list(X.columns))


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

data = pd.concat([X, y.rename("target")], axis=1)

before = len(data)

data = data.drop_duplicates()

after = len(data)

print(f"\nDuplicates removed: {before - after}")

X = data.drop(columns=["target"])
y = data["target"]


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 7. HYPERPARAMETER GRIDS
# ============================================================

param_grids = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l1", "l2"]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    },

    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },

    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.03, 0.05, 0.1]
    }
}


# ============================================================
# 8. TRAIN + TUNE
# ============================================================

results = []

best_model = None
best_model_name = None
best_accuracy = -1


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    model_best = grid.best_estimator_

    print("Best parameters:")
    print(grid.best_params_)

    # Prediction
    y_pred = model_best.predict(X_test)

    # Probability
    if hasattr(model_best, "predict_proba"):
        y_prob = model_best.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_best
        best_model_name = name


# ============================================================
# 9. RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n\nMODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

print("\nBest Model:")
print(best_model_name)

print(f"Best Test Accuracy: {best_accuracy:.4f}")


# ============================================================
# 10. CREATE MODEL DIRECTORY
# ============================================================

model_dir = os.path.join(
    "models",
    "heart_disease"
)

os.makedirs(model_dir, exist_ok=True)


# ============================================================
# 11. SAVE SCALER
# ============================================================

# The Streamlit app expects a separate scaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# ============================================================
# 12. IMPORTANT:
#     IF BEST MODEL IS A PIPELINE, EXTRACT THE ACTUAL MODEL
# ============================================================

if isinstance(best_model, Pipeline):

    actual_model = best_model.named_steps["model"]

else:

    actual_model = best_model


# ============================================================
# 13. TRAIN FINAL MODEL ON FULL DATA
# ============================================================

print("\nTraining final model on full dataset...")

X_scaled_full = scaler.fit_transform(X)

actual_model.fit(
    X_scaled_full,
    y
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    model_dir,
    "best_model.pkl"
)

scaler_path = os.path.join(
    model_dir,
    "scaler.pkl"
)

features_path = os.path.join(
    model_dir,
    "features.pkl"
)


joblib.dump(
    actual_model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    list(X.columns),
    features_path
)


# ============================================================
# 15. VERIFY SAVED MODEL
# ============================================================

print("\n" + "=" * 60)
print("VERIFYING SAVED MODEL")
print("=" * 60)

loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)
loaded_features = joblib.load(features_path)

print("Model:", type(loaded_model))
print("Model fitted:", hasattr(loaded_model, "coef_") or hasattr(loaded_model, "estimators_"))
print("Scaler:", type(loaded_scaler))
print("Features:", len(loaded_features))

# Test prediction
test_sample = X.iloc[[0]]

test_scaled = loaded_scaler.transform(test_sample)

test_prediction = loaded_model.predict(test_scaled)

print("Test prediction:", test_prediction)

print("\nFiles saved successfully:")
print(model_path)
print(scaler_path)
print(features_path)

print("\nDONE!")