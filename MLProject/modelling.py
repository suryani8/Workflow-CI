# modelling.py - MLProject

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report,
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# ARGUMENT PARSER
# Menerima input path dataset dari command line
parser = argparse.ArgumentParser()
parser.add_argument('--input_train', type=str, default='heart_preprocessing_train.csv')
parser.add_argument('--input_test',  type=str, default='heart_preprocessing_test.csv')
args = parser.parse_args()

# KONEKSI KE DAGSHUB
os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get('MLFLOW_TRACKING_USERNAME', 'suryani8')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get('MLFLOW_TRACKING_PASSWORD', 'eb138e8528c3a6a4be25f41f26ee5f8283f65d04')

tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'https://dagshub.com/suryani8/Eksperimen_SML_Suryani_apc367d6x0436.mlflow')
mlflow.set_tracking_uri(tracking_uri)

# Set experiment name agar run masuk ke experiment
mlflow.set_experiment("Heart Disease - CI Pipeline")

# LOAD DATA
# Dataset hasil preprocessing yang sudah siap digunakan untuk training
train_df = pd.read_csv(args.input_train)
test_df  = pd.read_csv(args.input_test)

train_df = train_df.drop(columns=['dataset'], errors='ignore')
test_df  = test_df.drop(columns=['dataset'], errors='ignore')

le = LabelEncoder()
binary_features = ['sex', 'fbs', 'exang']
for col in binary_features:
    if train_df[col].dtype == 'object':
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test  = test_df.drop(columns=['target'])
y_test  = test_df['target']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")

# HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV
# Mencari kombinasi hyperparameter terbaik untuk Random Forest
param_grid = {
    'n_estimators'     : [50, 100, 200],
    'max_depth'        : [None, 5, 10],
    'min_samples_split': [2, 5],
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"\nBest Parameters: {best_params}")

# MANUAL LOGGING
y_pred      = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

mlflow.log_param("n_estimators",      best_params['n_estimators'])
mlflow.log_param("max_depth",         best_params['max_depth'])
mlflow.log_param("min_samples_split", best_params['min_samples_split'])
mlflow.log_param("cv_folds",          5)
mlflow.log_param("random_state",      42)

mlflow.log_metric("accuracy",  acc)
mlflow.log_metric("precision", prec)
mlflow.log_metric("recall",    rec)
mlflow.log_metric("f1_score",  f1)
mlflow.log_metric("roc_auc",   roc_auc)

print(f"\n=== Hasil Evaluasi Model ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(classification_report(y_test, y_pred))

# Artefak 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Sakit', 'Sakit'],
            yticklabels=['Tidak Sakit', 'Sakit'])
plt.title('Confusion Matrix - CI Pipeline')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
mlflow.log_artifact('confusion_matrix.png')

# Artefak 2: Feature Importance
feat_imp = pd.Series(best_model.feature_importances_,
                     index=X_train.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Top 15 Feature Importance - CI Pipeline')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
mlflow.log_artifact('feature_importance.png')

# Artefak 3: ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CI Pipeline')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
mlflow.log_artifact('roc_curve.png')

# Log model
mlflow.sklearn.log_model(best_model, "random_forest_ci")
print("Model tersimpan ke MLflow.")
print("\nmodelling.py")
