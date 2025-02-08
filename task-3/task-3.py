import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Use LinearRegression for continuous data
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load the car data
car_data = pd.read_csv('car data.csv')

# Display the columns to understand the dataset
print(car_data.columns)

# Set up the inputs (features) and outputs (target)
outputs = car_data['Present_Price']
inputs = car_data.drop(columns=['Present_Price', 'Car_Name'], axis=1)  # Dropping 'Car_Name' too since it's not useful

# Perform One-Hot Encoding for categorical columns
inputs = pd.get_dummies(inputs, drop_first=True)  # drop_first=True to avoid multicollinearity

# Check the input columns after encoding
print(inputs.columns)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, random_state=42, test_size=0.3)

# Initialize the model (Linear Regression for continuous data)
model = LinearRegression()

# Fit the model on the training data
model.fit(x_train, y_train)

# Predict on the test data
y_pred = model.predict(x_test)

# Evaluate the model using R-squared and Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
