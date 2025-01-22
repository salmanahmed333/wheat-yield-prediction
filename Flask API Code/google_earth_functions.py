import ee
import calendar

# Initialize Google Earth Engine
ee.Authenticate() 
ee.Initialize(project="ee-salmanahmed10124")

# Function to extract area parameter from input KML file
def get_area(coordinates):
    ee_polygon = ee.Geometry.Polygon(coordinates)
    area_in_sq_m = ee_polygon.area().getInfo()
    return area_in_sq_m

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
        print(f"Indices for {year}-{month} were not calculated: {e}")
        return [None, None, None, None, None]
