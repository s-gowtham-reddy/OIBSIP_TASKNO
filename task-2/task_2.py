import kagglehub as hub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download dataset from kagglehub
unemployment_data_path = hub.dataset_download("gokulrajkmv/unemployment-in-india")
# print(unemployment_data_path)

unemp_data = pd.read_csv('unemp.csv')
# print(unemp_data.describe())

# Drop rows with all NaN values
unemp_data = unemp_data.dropna(axis=0, how='all')
# print(unemp_data.describe())

# Separate inputs (features) and outputs (target)
inputs = unemp_data.iloc[:, :-1]
outputs = unemp_data.iloc[:, -1]

# Print column names to ensure no mistakes with column names
# print(inputs.columns)

# Drop the 'Date' column from the 'inputs' DataFrame (make sure there is no leading space in column name)
inputs.drop(columns=" Date", inplace=True, axis=1)

# Apply One-Hot Encoding to categorical columns (such as 'State')
inputs = pd.get_dummies(inputs)

# Check the column names after encoding
# print(inputs.columns)

# Split the data into training and testing sets (60% train, 40% test)
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, random_state=42, test_size=0.4)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy of the model is {acc}")
print(f"Classification report ::\n{class_report}")
