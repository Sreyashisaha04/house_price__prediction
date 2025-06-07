# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Create the dataset
data = {
    'bedrooms': [3, 2, 4, 3, 3, 2, 4, 3, 4, 5],
    'size': [1500, 1200, 2000, 1600, 1700, 1300, 2100, 1600, 2500, 3000],
    'age': [10, 5, 20, 15, 10, 5, 25, 10, 30, 50],
    'price': [300000, 250000, 400000, 350000, 340000, 280000, 410000, 360000, 450000, 500000]
}

# Load the data into a DataFrame
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Split features and target
X = df[['bedrooms', 'size', 'age']]
y = df['price']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid(True)
plt.show()

# Predict new data
new_data = pd.DataFrame({
    'bedrooms': [3, 4],
    'size': [1800, 2200],
    'age': [8, 12]
})
new_data_scaled = scaler.transform(new_data)
new_predictions = model.predict(new_data_scaled)

print("\n🏡 Predicted Prices for New Data:\n", new_predictions)
