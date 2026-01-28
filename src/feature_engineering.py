import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
print("Loading dataset...")
try:
    df = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: creditcard.csv not found.")
    exit()

# 2. Minimal Feature Engineering
print("\nPerforming Feature Engineering...")

# log_amount: Log transformation to handle skewed Amount distribution
# Adding 1e-5 to avoid log(0) if Amount is 0 (though normally transactions > 0 or 0 for verification)
df['log_amount'] = np.log1p(df['Amount'])

# amount_zscore: Standardizing Amount
scaler = StandardScaler()
df['amount_zscore'] = scaler.fit_transform(df[['Amount']])

# is_high_amount: Binary flag for high transactions (> 95th percentile)
threshold_95 = df['Amount'].quantile(0.95)
df['is_high_amount'] = (df['Amount'] > threshold_95).astype(int)
print(f"High amount threshold (95th percentile): {threshold_95:.2f}")

# amount_per_time: Interaction feature (Amount / Time) which might capture velocity/rate anomalies
# Adding 1 to Time to avoid division by zero
df['amount_per_time'] = df['Amount'] / (df['Time'] + 1)

print("New features created: log_amount, amount_zscore, is_high_amount, amount_per_time")

# 3. Prepare for Feature Selection
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Feature Selection (Random Forest)
print("\nTraining Random Forest for Feature Importance (this may take a moment)...")
# Using a limited depth and estimators for speed in this assessment script, 
# but sufficient for importance ranking
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5. Extract and Print Feature Importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance Ranking ---")
print(importances.head(20))  # Top 20 features

# Highlight new features
new_features = ['log_amount', 'amount_zscore', 'is_high_amount', 'amount_per_time']
print("\n--- New Feature Performance ---")
print(importances[importances['Feature'].isin(new_features)])

# Optional: Suggest features to keep (e.g., top 15 + new features if relevant)
top_features = importances.head(15)['Feature'].tolist()
print("\nSuggested Top 15 Features:", top_features)
