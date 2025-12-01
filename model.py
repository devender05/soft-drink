# train_profit_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Load your real CSV file
df = pd.read_csv("soft_drink_sales.csv")  # Change filename if needed

# Create target: 1 = Profitable, 0 = Loss
df['profitable'] = (df['Profit'] > 0).astype(int)

# Feature engineering
df['is_coke'] = df["Company"].str.lower().str.contains("coca", na=False).astype(int)
df['category_length'] = df["Category"].str.len()

# Use only numeric features
features = ["Units Sold", "Revenue", "Cost of Goods Sold", "is_coke", "category_length"]
X = df[features]
y = df['profitable']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Save model
import os
os.makedirs("models", exist_ok=True)
with open("models/profit_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"Model saved! Accuracy: {model.score(X_test, y_test):.1%}")
print(f"Profitable in data: {y.mean():.1%} of rows")