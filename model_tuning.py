import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# 1. Pipeline Reconstruction (Data Load + Feature Engineering + Split + SMOTE)
print("Setting up pipeline (Dataset -> Features -> Split -> SMOTE)...")

# Load
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    print("Error: creditcard.csv not found.")
    exit()

# Feature Engineering (The 34 Features)
# "Minimal but High-Impact" Features
df['log_amount'] = np.log1p(df['Amount'])
df['amount_per_time'] = df['Amount'] / (df['Time'] + 1)
threshold_95 = df['Amount'].quantile(0.95)
df['is_high_amount'] = (df['Amount'] > threshold_95).astype(int)
scaler = StandardScaler()
df['amount_zscore'] = scaler.fit_transform(df[['Amount']])

# Context + PCA Features (V1-V28) + Amount/Time
# Keeping ALL features as XGBoost handles irrelevance well
X = df.drop("Class", axis=1)
y = df["Class"]

print(f"Using all {X.shape[1]} features for training.")

# Split (Stratified to keep test set realistic)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE (Balancing ONLY the training set)
print("Applying SMOTE balancing...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Training shape: {X_train_resampled.shape}")
print(f"Test shape: {X_test.shape}")

# 2. Hyperparameter Tuning Setup
print("\n--- Starting Model Tuning ---")

# A. Random Forest
rf_params = {
    'n_estimators': [100, 200], # Trees
    'max_depth': [10, 20, None], # Complexity
    'min_samples_split': [2, 5, 10], # Overfitting control
    'max_features': ['sqrt', 'log2'] 
}

# B. XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [1, 10] # Helps with any remaining imbalance nuances
}

# 3. Randomized Search Execution
def tune_model(model, params, name):
    print(f"\nTuning {name}...")
    # RandomizedSearchCV is faster than GridSearch for exploratory tuning
    search = RandomizedSearchCV(
        model, 
        params, 
        n_iter=2, # Reduced for speed (was 5)
        scoring='f1', # Optimization target: F1 Score
        cv=2, # Reduced for speed (was 3)
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train_resampled, y_train_resampled)
    print(f"Best {name} Params: {search.best_params_}")
    return search.best_estimator_

# Tune RF
best_rf = tune_model(RandomForestClassifier(random_state=42), rf_params, "Random Forest")

# Tune XGB
best_xgb = tune_model(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), xgb_params, "XGBoost")

# 4. Final Evaluation on Test Set (Unseen Real-World Data)
def evaluate(model, name):
    print(f"\n--- {name} Evaluation (Test Set) ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 5. Ensemble Learning (Voting Classifier)
from sklearn.ensemble import VotingClassifier

print("\n--- Training Voting Ensemble (RF + XGB) ---")
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf), ('xgb', best_xgb)],
    voting='soft' 
)
voting_clf.fit(X_train_resampled, y_train_resampled)

evaluate(voting_clf, "Voting Ensemble (RF + XGB)")

