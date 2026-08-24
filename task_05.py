import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression

# 1. Dataset Simulation / Food Feature Embeddings & Calorie Target
# Simulating food image visual feature embeddings (128-d vector) to predict calorie count
X, y = make_regression(
    n_samples=1500,
    n_features=128,
    n_informative=60,
    noise=10.0,
    random_state=42
)

# Scaling target values to realistic calorie ranges (50 to 800 kcal)
y = np.interp(y, (y.min(), y.max()), (50, 800))

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Regression Model for Calorie Estimation
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# 4. Evaluate Model Performance
y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kcal")
print(f"R^2 Score: {r2:.4f}")

# Sample Predictions
print("\n--- Sample Calorie Predictions (Actual vs Predicted) ---")
sample_df = pd.DataFrame({'Actual Calorie (kcal)': y_test[:5], 'Predicted Calorie (kcal)': y_pred[:5]})
print(sample_df.round(2))
