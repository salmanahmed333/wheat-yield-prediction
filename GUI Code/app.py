import streamlit as st # For GUI
from pykml import parser # For parsing KML file
import ee # Google Earth Engine (GEE) API
from datetime import datetime, timedelta # For dealing with datetime data
import calendar  # For determining the correct number of days in a month
import requests
import pickle
import numpy as np

ee.Initialize(project="ee-salmanahmed10124") # Using a cloud project for connection with GEE API

# Function to extract coordinates of a field polygon inside KML file
def extract_coordiantes(field_file):
    root = parser.parse(field_file).getroot()
    coordinates = []
    for placemark in root.Document.Placemark:
        if placemark.Polygon is not None:
            coords = placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip()
            coords = coords.split()
            for point in coords:
                long, lat, _ = point.split(",")
                coordinates.append([float(long), float(lat)])
    return coordinates

# Helper function to get monthly filtered Sentinel-2 imagery
def get_sentinel2_monthly_image(coordinates, start_date, end_date):
    polygon = ee.Geometry.Polygon(coordinates)
    collection = (
        ee.ImageCollection("COPERNICUS/S2")
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # Cloud masking
    )
    return collection.median().clip(polygon)

# Monthly indices calculation using GEE
def calculate_indices(coordinates, year, month):
    try:
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"
        image = get_sentinel2_monthly_image(coordinates, start_date, end_date)
        # Calculate all indices
        indices = {
            "NDVI": image.normalizedDifference(["B8", "B4"]).rename("NDVI"),  # NIR: B8, Red: B4
            "GNDVI": image.normalizedDifference(["B8", "B3"]).rename("GNDVI"),  # NIR: B8, Green: B3
            "NDMI": image.normalizedDifference(["B8", "B11"]).rename("NDMI"),  # NIR: B8, SWIR: B11
            "SAVI": image.select("B8").subtract(image.select("B4"))
                    .divide(image.select("B8").add(image.select("B4")).add(0.5))
                    .multiply(1.5).rename("SAVI"),  # NIR: B8, Red: B4
            "NDRE": image.normalizedDifference(["B8", "B5"]).rename("NDRE")  # NIR: B8, Red Edge: B5
        }
        # Compute mean values for each index
        mean_values = []
        for index_name, index_image in indices.items():
            mean_value = index_image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=image.geometry(), scale=30
            ).getInfo().get(index_name, None)  # Retrieve the value by the band's name
            mean_values.append(mean_value)
        return mean_values
    except Exception as e:
        st.write(f"Indices for {year}-{month} were not calculated")
        return [None, None, None, None, None]

def fetch_weather_data(api_key, location, year, month):
    """
    Fetch weather data for a given year and month.
    """
    base_url = "http://api.weatherapi.com/v1/history.json"
    # Start and end dates of the specified month
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, calendar.monthrange(year, month)[1])
    current_date = start_date
    delta = timedelta(days=1)

    # Accumulators for averaging
    weather_accum = {
        "avgtemp_c": [],
        "maxwind_kph": [],
        "totalprecip_mm": 0,
        "avghumidity": [],
        "sunshine_hours": []
    }

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        params = {"key": api_key, "q": location, "dt": date_str}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract forecast details
            forecast = data.get("forecast", {}).get("forecastday", [])[0]
            astro = forecast.get("astro", {})
            day = forecast.get("day", {})

            # Calculate sunshine hours
            sunrise = astro.get("sunrise")
            sunset = astro.get("sunset")
            if sunrise and sunset:
                sunshine_duration = (
                    datetime.strptime(sunset, "%I:%M %p") - datetime.strptime(sunrise, "%I:%M %p")
                ).total_seconds() / 3600
            else:
                sunshine_duration = 0

            # Accumulate daily values
            weather_accum["avgtemp_c"].append(day.get("avgtemp_c", 0))
            weather_accum["maxwind_kph"].append(day.get("maxwind_kph", 0))
            weather_accum["totalprecip_mm"] += day.get("totalprecip_mm", 0)
            weather_accum["avghumidity"].append(day.get("avghumidity", 0))
            weather_accum["sunshine_hours"].append(sunshine_duration)

        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")

        current_date += delta

    # Calculate averages
    averaged_data = [
        sum(weather_accum["avgtemp_c"]) / len(weather_accum["avgtemp_c"]) if weather_accum["avgtemp_c"] else None,
        sum(weather_accum["avghumidity"]) / len(weather_accum["avghumidity"]) if weather_accum["avghumidity"] else None,
        weather_accum["totalprecip_mm"],
        sum(weather_accum["maxwind_kph"]) / len(weather_accum["maxwind_kph"]) if weather_accum["maxwind_kph"] else None,
        sum(weather_accum["sunshine_hours"]) / len(weather_accum["sunshine_hours"]) if weather_accum["sunshine_hours"] else None
    ]

    return averaged_data
# Function to extract area parameter from input KML file
def get_area(coordinates):
    ee_polygon = ee.Geometry.Polygon(coordinates)
    area_in_sq_m = ee_polygon.area().getInfo()
    return area_in_sq_m

def mean_coordinates(coordinates):
    long = 0
    lat = 0
    for point in coordinates:
        long += point[0]
        lat += point[1]
    location = f"{lat/len(coordinates)},{long/len(coordinates)}"
    return location

# Define the prediction function
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

# Intro section
st.title("Wheat Crop Yield Prediction Model")
st.subheader("Crop Cycle Duration")

# Dropdowns for start date
st.write("Select Starting Date")
col1, col2 = st.columns(2)  # Create two columns for better alignment
with col1:
    start_year = st.selectbox("Start Year", list(range(2016, 2025)), index=0)
with col2:
    start_month = st.selectbox("Start Month", list(range(1, 13)), index=0)

# Dropdowns for end date
st.write("Select Ending Date")
col3, col4 = st.columns(2)  # Create two columns for better alignment

# Dynamically filter end year and end month based on start date
with col3:
    valid_end_years = list(range(start_year, 2025))  # Years >= start year
    end_year = st.selectbox("End Year", valid_end_years, index=0)

with col4:
    if end_year == start_year:
        valid_end_months = list(range(start_month, 13))  # Months > start month
    else:
        valid_end_months = list(range(1, 13))
    end_month = st.selectbox("End Month", valid_end_months, index=0)

# File uploader for field data
st.subheader("Wheat Field Information")
district = st.selectbox("Select District", list(["Naushahro Feroze, Sindh", "Rahim Yar Khan, Punjab"]), index=0)
field_file = st.file_uploader("Upload a KML file of the field to predict the wheat yield", type=["kml"])

# "Predict Yield" button
predict_button = st.button("Predict Yield", key="predict", help="Click to predict wheat yield")

if predict_button:
    message = st.empty()
    if field_file is None:
        message.write("Please upload a KML fie first!")
    else:
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        message.write("Parsing file data...")
        field_coordinates = extract_coordiantes(field_file)
        current_date = start_date
        message.write("Extracting area value...")
        # Calculate and display area
        area_in_sq_m = get_area(field_coordinates)

        api_key = "efb3685cfdd64117aa0102758242911"
        location = mean_coordinates(field_coordinates)
        prediction_data = []
        # Monthly indices calculations
        while current_date <= end_date:
            message.write(f"Fetching weather data for {current_date.year}-{current_date.month}...")
            weather = fetch_weather_data(api_key, location, current_date.year, current_date.month)
            message.write(f"Calculating indices values for {current_date.year}-{current_date.month}...")
            # Get index values for the current month
            indices = calculate_indices(field_coordinates, current_date.year, current_date.month)
            prediction_data.append(weather+indices)
            # Update to the next month
            if current_date.month == 12:  # If it's December, roll over to January
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        # st.write(prediction_data)
        # Prediction and Results
        message.write("Predicting wheat yield using ML Model...")
        message.write(f"Prediction Duration: From {start_date.year}-{start_date.month} To {end_date.year}-{end_date.month}")
        st.subheader("Results from Linear Regression Model")
        if district == "Naushahro Feroze, Sindh":
            linear_regression_sindh = "linear_regression_sindh.pkl"
            with open(linear_regression_sindh, "rb") as file:
                lr_model_sindh = pickle.load(file)
            mean_yield = predict_mean_yield(lr_model_sindh, prediction_data)
        elif district == "Rahim Yar Khan, Punjab":
            linear_regression_punjab = "linear_regression_punjab.pkl"
            with open(linear_regression_punjab, "rb") as file:
                lr_model_punjab = pickle.load(file)
            mean_yield = predict_mean_yield(lr_model_punjab, prediction_data)
        else:
            mean_yield = None
            st.write("Invalid district selection.")

        if mean_yield is not None:
            st.write(f"Predicted Wheat Yield: {mean_yield:.4f} tonnes per hectare")
        else:
            st.write("Unable to predict yield. Please verify inputs.")
        area_in_ha = area_in_sq_m/10000
        area_in_acres = area_in_sq_m/4047
        st.write(f"Total Area: {round(area_in_acres, 4)} acres or {round(area_in_ha, 4)} hectares")
