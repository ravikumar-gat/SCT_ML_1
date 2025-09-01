# --- House Price Prediction using Linear Regression ---

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# Step 2: Load Dataset
file_path = "Housing.csv"   # replace with your dataset filename
data = pd.read_csv(file_path)

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:\n", data.head())

# Step 3: Select Features and Target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions & Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R² Score:", r2)

# Step 7: Example Prediction
example = pd.DataFrame({'area':[3000], 'bedrooms':[3], 'bathrooms':[2]})
predicted_price = model.predict(example)
print("\nPredicted Price for Example House:", predicted_price[0])

# Step 8: Visualizations

# 8.1 Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# 8.2 Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7, color="green")
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# 8.3 3D Visualizations (Bedrooms & Bathrooms)

fig = plt.figure(figsize=(16,7))

# Plot 1: Area vs Bedrooms vs Price
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlabel('Area (sq ft)')
ax1.set_ylabel('Bedrooms')
ax1.set_zlabel('Predicted Price')
ax1.set_title("Area vs Bedrooms vs Price")

ax1.scatter(X['area'], X['bedrooms'], y, c='blue', marker='o', alpha=0.5)

area_range = np.linspace(X['area'].min(), X['area'].max(), 20)
bedrooms_range = np.linspace(X['bedrooms'].min(), X['bedrooms'].max(), 20)
area_grid, bedrooms_grid = np.meshgrid(area_range, bedrooms_range)

bathrooms_avg = int(X['bathrooms'].mean())
Z_pred1 = model.predict(np.c_[area_grid.ravel(), bedrooms_grid.ravel(), np.full(area_grid.ravel().shape, bathrooms_avg)])
Z_pred1 = Z_pred1.reshape(area_grid.shape)

ax1.plot_surface(area_grid, bedrooms_grid, Z_pred1, color='red', alpha=0.5)

# Plot 2: Area vs Bathrooms vs Price
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlabel('Area (sq ft)')
ax2.set_ylabel('Bathrooms')
ax2.set_zlabel('Predicted Price')
ax2.set_title("Area vs Bathrooms vs Price")

ax2.scatter(X['area'], X['bathrooms'], y, c='green', marker='o', alpha=0.5)

bathrooms_range = np.linspace(X['bathrooms'].min(), X['bathrooms'].max(), 20)
area_grid, bathrooms_grid = np.meshgrid(area_range, bathrooms_range)

bedrooms_avg = int(X['bedrooms'].mean())
Z_pred2 = model.predict(np.c_[area_grid.ravel(), np.full(area_grid.ravel().shape, bedrooms_avg), bathrooms_grid.ravel()])
Z_pred2 = Z_pred2.reshape(area_grid.shape)

ax2.plot_surface(area_grid, bathrooms_grid, Z_pred2, color='orange', alpha=0.5)

plt.tight_layout()
plt.show()
