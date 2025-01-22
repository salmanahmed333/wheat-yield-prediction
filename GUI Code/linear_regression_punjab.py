import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

# 1. Read the dataset
data = pd.read_csv("rahim_yar_khan_data.csv")

# 2. Separate features (X) and target variable (y)
# Exclude non-numeric and target columns like 'name', 'month', 'year', and 'yield'
X = data.drop(columns=["name", "month", "year", "area", "yield"])
y = data["yield"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 6. Save the trained model into a pickle file
pickle_filename = "linear_regression_punjab.pkl"
with open(pickle_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as {pickle_filename}")

