# predict_house.py


import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from colorama import init, Fore, Style
import sys
init(autoreset=True)


# Load dataset
data = pd.read_csv("Housing.csv")

# Prepare features and target
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Optional: Save model for later use
pickle.dump(model, open("house_price_model.pkl", "wb"))

# Model summary
def model_summary():
    print(Fore.CYAN + Style.BRIGHT + "\nModel Summary:")
    print(Fore.YELLOW + f"  Coefficients: Area={model.coef_[0]:.2f}, Bedrooms={model.coef_[1]:.2f}, Bathrooms={model.coef_[2]:.2f}")
    print(Fore.YELLOW + f"  Intercept: {model.intercept_:.2f}")
    r2 = model.score(X, y)
    print(Fore.YELLOW + f"  R² Score: {r2:.4f}\n")


# Function for prediction
def predict_house_price(area, bedrooms, bathrooms):
    example = pd.DataFrame({'area':[area], 'bedrooms':[bedrooms], 'bathrooms':[bathrooms]})
    price = model.predict(example)[0]
    return round(price, 2)


# ASCII Art Title
def print_title():
    print(Fore.GREEN + Style.BRIGHT + """
  _    _                         _     _            _             
 | |  | |                       | |   | |          | |            
 | |__| | ___  _ __ ___   ___   | |__ | | ___   ___| | _____ _ __ 
 |  __  |/ _ \| '_ ` _ \ / _ \  | '_ \| |/ _ \ / __| |/ / _ \ '__|
 | |  | | (_) | | | | | |  __/  | |_) | | (_) | (__|   <  __/ |   
 |_|  |_|\___/|_| |_| |_|\___|  |_.__/|_|\___/ \___|_|\_\___|_|   
    """ + Style.RESET_ALL)
    print(Fore.MAGENTA + Style.BRIGHT + "Welcome to the House Price Prediction Tool!\n" + Style.RESET_ALL)

# Input validation
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
    model_summary()
    while True:
        print(Fore.BLUE + Style.BRIGHT + "Menu:")
        print(Fore.YELLOW + "  1. Predict House Price")
        print(Fore.YELLOW + "  2. Exit\n")
        choice = input(Fore.CYAN + "Enter your choice (1/2): " + Style.RESET_ALL)
        if choice == '1':
            area = get_float("Enter area (sq ft): ", min_value=100)
            bedrooms = get_int("Enter number of bedrooms: ", min_value=1)
            bathrooms = get_int("Enter number of bathrooms: ", min_value=1)
            predicted_price = predict_house_price(area, bedrooms, bathrooms)
            print(Fore.GREEN + Style.BRIGHT + f"\n💰 Predicted House Price: ${predicted_price:,.2f}\n")
        elif choice == '2':
            print(Fore.MAGENTA + "Thank you for using the House Price Prediction Tool! Goodbye!\n")
            sys.exit(0)
        else:
            print(Fore.RED + "Invalid choice. Please enter 1 or 2.\n")

if __name__ == "__main__":
    main_menu()
