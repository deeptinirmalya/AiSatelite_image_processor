"""
Google Earth Engine integration for satellite imagery timeline
"""
import os
from dotenv import load_dotenv
import ee
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def initialize_earth_engine():
    """Initialize Google Earth Engine with service account credentials"""
    try:
        # Get credentials from environment
        project_id = os.getenv('GEE_PROJECT_ID')
        client_email = os.getenv('GEE_CLIENT_EMAIL')
        private_key = os.getenv('GEE_PRIVATE_KEY', '').replace('\\n', '\n')
        
        if not all([project_id, client_email, private_key]):
            print("Warning: Google Earth Engine credentials not found in .env file")
            return False
            
        # Basic validation: Service Account keys must be RSA private keys (long strings with BEGIN/END markers)
        # If it's an API Key (starts with AIza), it's the wrong type of credential for this method.
        if private_key.startswith('AIza'):
            print("❌ Error: GEE_PRIVATE_KEY appears to be a Google API Key, but a Service Account Private Key is required for Earth Engine initialization.")
            return False
        
        if "-----BEGIN PRIVATE KEY-----" not in private_key:
            print("❌ Error: GEE_PRIVATE_KEY is missing the '-----BEGIN PRIVATE KEY-----' marker.")
            return False
        
        # Create credentials
        credentials = ee.ServiceAccountCredentials(client_email, key_data=private_key)
        ee.Initialize(credentials, project=project_id)
        print("✅ Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize Google Earth Engine: {e}")
        return False

def get_satellite_image_url(lat, lon, year, zoom=12):
    """
    Get satellite image URL from Google Earth Engine for a specific location and year.
    Uses multi-sensor fusion: Landsat for older history, Sentinel-2 for recent high-res.
    """
    try:
        # Define the point of interest
        point = ee.Geometry.Point([lon, lat])
        
        # Define date range for the year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        # Choose collection based on year
        sensor = ""
        resolution = ""
        if year >= 2016:
            # Sentinel-2 (High Resolution)
            sensor = "Sentinel-2"
            resolution = "10m/px"
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                         .filterBounds(point)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) # Stricter cloud filter
                         .select(['B4', 'B3', 'B2']))
            vis_params = {'min': 0, 'max': 3500, 'gamma': 1.2} # Improved contrast
        elif year >= 2013:
            # Landsat 8
            sensor = "Landsat 8"
            resolution = "30m/px"
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterBounds(point)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 10))
                         .select(['SR_B4', 'SR_B3', 'SR_B2']))
            vis_params = {'min': 7000, 'max': 19000, 'gamma': 1.1}
        else:
            # Landsat 7
            sensor = "Landsat 7"
            resolution = "30m/px"
            collection = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
                         .filterBounds(point)
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 10))
                         .select(['SR_B3', 'SR_B2', 'SR_B1']))
            vis_params = {'min': 7000, 'max': 19000, 'gamma': 1.1}
            
        # Get median composite to remove transient clouds
        image = collection.median()
        
        if collection.size().getInfo() == 0:
            return None

        # Get high-resolution tile URL for the map
        # This provides crystal clear imagery regardless of zoom level
        map_id_dict = ee.Image(image).getMapId(vis_params)
        tile_url = map_id_dict['tile_fetcher'].url_format
        
        # Prepare thumbnail for the sidebar
        region = point.buffer(2000).bounds()
        thumb_params = {
            'region': region,
            'dimensions': '1024',
            'format': 'png'
        }
        thumb_params.update(vis_params)
        thumb_url = image.getThumbURL(thumb_params)
        
        # Exact geographic math for bounds
        lat_step = 2000 / 111320.0
        import math
        lon_step = 2000 / (111320.0 * math.cos(math.radians(lat)))
        
        bounds = [[lat - lat_step, lon - lon_step], [lat + lat_step, lon + lon_step]]
        
        print(f"✅ Generated sharp tiles and thumb for {year}")
        
        return {
            "url": thumb_url,
            "tile_url": tile_url,
            "bounds": bounds,
            "metadata": {
                "sensor": sensor,
                "resolution": resolution,
                "provider": "Copernicus/USGS via GEE"
            }
        }
    except Exception as e:
        print(f"❌ Error getting satellite image for {year}: {e}")
        return None

from concurrent.futures import ThreadPoolExecutor

def get_timeline_images(lat, lon):
    """
    Get satellite images for historical timeline (last 5 years)
    Uses multi-threading for blazing fast concurrent loading
    """
    timeline = {}
    current_year = datetime.now().year
    # Limit to last 5 years as requested
    years = list(range(current_year - 5, current_year + 1, 1))
    
    # Use ThreadPoolExecutor to fetch all years in parallel
    def fetch_year(year):
        try:
            return year, get_satellite_image_url(lat, lon, year)
        except Exception as e:
            print(f"Parallel fetch error for {year}: {e}")
            return year, None

    print(f"🚀 Initializing parallel uplink for {len(years)} orbital sectors...")
    with ThreadPoolExecutor(max_workers=len(years)) as executor:
        results = list(executor.map(fetch_year, years))
    
    for year, data in results:
        if data:
            timeline[year] = data
            
    return timeline
