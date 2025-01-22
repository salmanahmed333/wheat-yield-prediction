import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

# 1. Read the dataset
data = pd.read_csv("naushahro_feroze_data.csv")

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
pickle_filename = "linear_regression_sindh.pkl"
with open(pickle_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as {pickle_filename}")

# # Prediction data
# data = [
# [50.87, 19.33225806451613, 18.12258064516129],
# [60.65, 23.70967741935484, 16.903225806451612],
# [1.8, 1.1, 0],
# [13.59, 10.883870967741935, 10.551612903225807],
# [10.83, 10.47258064516129, 10.65752688172043],
# [0.257, 0.26285361313072353, 0.6448687116759124],
# [0.306, 0.16069546598788384, 0.5185312528250626],
# [-0.03, 0.09684059662225693, 0.4231883725605143],
# [-0.168, 0.1477698122624971, 0.5082681538575035],
# [0.385, 0.39420983565090245, 0.9671811256065631]
# ]

# yields = []
# for i in range(len(data[0])):
#     feature_vector = [data[j][i] for j in range(len(data))]
#     feature_array = np.array(feature_vector).reshape(1, -1)
#     yield_prediction = model.predict(feature_array)[0]
#     yields.append(yield_prediction)
# print(yields)

