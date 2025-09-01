# house_price_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from colorama import init, Fore, Style
import sys
import pickle
init(autoreset=True)

# Load Dataset
file_path = "Housing.csv"  # Use the correct dataset filename
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(Fore.RED + f"Dataset file '{file_path}' not found. Please check the filename.")
    sys.exit(1)

# Prepare features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model for later use
pickle.dump(model, open("house_price_model.pkl", "wb"))

# Model summary
mse = mean_squared_error(y_test, model.predict(X_test))
rmse = np.sqrt(mse)
r2 = r2_score(y_test, model.predict(X_test))

def model_summary():
    print(Fore.CYAN + Style.BRIGHT + "\nModel Summary:")
    print(Fore.YELLOW + f"  Coefficients: Area={model.coef_[0]:.2f}, Bedrooms={model.coef_[1]:.2f}, Bathrooms={model.coef_[2]:.2f}")
    print(Fore.YELLOW + f"  Intercept: {model.intercept_:.2f}")
    print(Fore.YELLOW + f"  RMSE: {rmse:,.2f}")
    print(Fore.YELLOW + f"  R² Score: {r2:.4f}\n")

def show_data_info():
    print(Fore.GREEN + Style.BRIGHT + f"\nDataset Shape: {data.shape}")
    print(Fore.GREEN + Style.BRIGHT + f"\nFirst 5 Rows:\n{data.head()}\n")

def show_visualizations():
    # 1. Actual vs Predicted
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    plt.show()

    # 2. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="green")
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.show()

    # 3. 3D Visualizations
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

def predict_house_price(area, bedrooms, bathrooms):
    example = pd.DataFrame({'area':[area], 'bedrooms':[bedrooms], 'bathrooms':[bathrooms]})
    price = model.predict(example)[0]
    return round(price, 2)

def print_title():
    print(Fore.GREEN + Style.BRIGHT + """
  _    _                         _     _            _             
 | |  | |                       | |   | |          | |            
 | |__| | ___  _ __ ___   ___   | |__ | | ___   ___| | _____ _ __ 
 |  __  |/ _ \\| '_ ` _ \\ / _ \\  | '_ \\| |/ _ \\ / __| |/ / _ \\ '__|
 | |  | | (_) | | | | | |  __/  | |_) | | (_) | (__|   <  __/ |   
 |_|  |_|\\___/|_| |_| |_|\\___|  |_.__/|_|\\___/ \\___|_|\\_\\___|_|   
    """ + Style.RESET_ALL)
    print(Fore.MAGENTA + Style.BRIGHT + "Welcome to the House Price Prediction Tool!\n" + Style.RESET_ALL)

def get_float(prompt, min_value=0):
    while True:
        try:
            value = float(input(Fore.CYAN + prompt + Style.RESET_ALL))
            if value < min_value:
                print(Fore.RED + f"Value must be at least {min_value}.")
                continue
            return value
        except ValueError:
            print(Fore.RED + "Please enter a valid number.")

def get_int(prompt, min_value=0):
    while True:
        try:
            value = int(input(Fore.CYAN + prompt + Style.RESET_ALL))
            if value < min_value:
                print(Fore.RED + f"Value must be at least {min_value}.")
                continue
            return value
        except ValueError:
            print(Fore.RED + "Please enter a valid integer.")

def main_menu():
    print_title()
    show_data_info()
    model_summary()
    while True:
        print(Fore.BLUE + Style.BRIGHT + "Menu:")
        print(Fore.YELLOW + "  1. Predict House Price")
        print(Fore.YELLOW + "  2. Show Visualizations")
        print(Fore.YELLOW + "  3. Exit\n")
        choice = input(Fore.CYAN + "Enter your choice (1/2/3): " + Style.RESET_ALL)
        if choice == '1':
            area = get_float("Enter area (sq ft): ", min_value=100)
            bedrooms = get_int("Enter number of bedrooms: ", min_value=1)
            bathrooms = get_int("Enter number of bathrooms: ", min_value=1)
            predicted_price = predict_house_price(area, bedrooms, bathrooms)
            print(Fore.GREEN + Style.BRIGHT + f"\n💰 Predicted House Price: ${predicted_price:,.2f}\n")
        elif choice == '2':
            show_visualizations()
        elif choice == '3':
            print(Fore.MAGENTA + "Thank you for using the House Price Prediction Tool! Goodbye!\n")
            sys.exit(0)
        else:
            print(Fore.RED + "Invalid choice. Please enter 1, 2, or 3.\n")

if __name__ == "__main__":
    main_menu()
