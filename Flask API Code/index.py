# Importing required python libraries
from flask import Flask, request, jsonify
from datetime import datetime
import pickle
import numpy as np
# Importing functions from .py files
import google_earth_functions
import weather_api_functions

# Function to predict mean yield using ML model
def predict_mean_yield(model, data):
    yields = []
    columns = ["temp", "humid", "precip", "wind_speed", "sunshine", "ndvi", "gndvi", "ndmi", "savi", "ndre"]
    for feature_vector in data:
        if None in feature_vector:
            continue
        feature_df = pd.DataFrame([feature_vector], columns=columns)  # Convert to DataFrame
        yield_prediction = model.predict(feature_df)[0]
        yields.append(yield_prediction)
    return np.mean(yields)

# Flask API
app = Flask(__name__)
@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    # Extract data
    start_year = int(data["start_year"])
    start_month = int(data["start_month"])
    end_year = int(data["end_year"])
    end_month = int(data["end_month"])
    district = data["district"]
    coordinates = data["coordinates"]
    coordinates = [[y, x] for x, y in coordinates]
    area_in_sq_m = google_earth_functions.get_area(coordinates)
    
    # Prepare for monthly data processing
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)
    current_date = start_date
    
    # Fetch and calculate weather data and indices
    prediction_data = []
    api_key = "efb3685cfdd64117aa0102758242911"
    location = weather_api_functions.mean_coordinates(coordinates)
    
    while current_date <= end_date:
        weather = weather_api_functions.fetch_weather_data(api_key, location, current_date.year, current_date.month)
        indices = google_earth_functions.calculate_indices(coordinates, current_date.year, current_date.month)
        prediction_data.append(weather + indices)
        
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    # Load model and make predictions
    if district == "Naushahro Feroze, Sindh":
        with open("linear_regression_sindh.pkl", "rb") as file:
            model = pickle.load(file)
    else:
        with open("linear_regression_punjab.pkl", "rb") as file:
            model = pickle.load(file)

    mean_yield = predict_mean_yield(model, prediction_data)

    # Return the result in JSON format
    return jsonify({
        "predicted_yield_in_tonnes_per_hectare": mean_yield,
        "area_in_sq_m": area_in_sq_m
    })

if __name__ == "__main__":
    app.run(debug=True)
