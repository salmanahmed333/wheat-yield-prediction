from datetime import datetime, timedelta
import calendar
import requests

# Function to calculate mean location coordinates to extract its weather data from API
def mean_coordinates(coordinates):
    long = 0
    lat = 0
    for point in coordinates:
        long += point[0]
        lat += point[1]
    location = f"{lat/len(coordinates)},{long/len(coordinates)}"
    return location

# Function to fetch monthly weather data from WeatherAPI
def fetch_weather_data(api_key, location, year, month):
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