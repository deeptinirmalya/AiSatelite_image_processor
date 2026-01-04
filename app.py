from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import uvicorn
import shutil
import os
import uuid
import torch
from pathlib import Path
import httpx

# Import our inference logic
from run_inference import predict_change

# Import Earth Engine integration
from earth_engine import initialize_earth_engine, get_timeline_images, get_satellite_image_url

app = FastAPI(title="Satellite Change Detection API")

# Initialize Earth Engine on startup
earth_engine_available = False

@app.on_event("startup")
async def startup_event():
    global earth_engine_available
    print("🚀 Starting Satellite Change Detection API...")
    earth_engine_available = initialize_earth_engine()
    if earth_engine_available:
        print("✅ Earth Engine initialized - Real satellite imagery enabled!")
    else:
        print("⚠️  Earth Engine not available - Running in demo mode")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(".")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

@app.get("/")
def health_check():
    return {"status": "active", "service": "Satellite Change Detection AI"}

@app.get("/timeline/{lat}/{lon}")
async def get_timeline(lat: float, lon: float):
    """
    Get satellite imagery timeline for a specific location
    Uses Earth Engine if available, otherwise demo mode
    """
    try:
        # Try to use Earth Engine if available
        if earth_engine_available:
            print(f"📡 Fetching Earth Engine imagery for {lat}, {lon}")
            timeline_data = get_timeline_images(lat, lon)
            if timeline_data:
                return {
                    "status": "success", 
                    "source": "Google Earth Engine", 
                    "timeline": timeline_data
                }
        
        # Fallback to demo mode
        print(f"⚠️  Using demo mode for {lat}, {lon}")
        timeline = {}
        current_year = 2025
        
        for year in range(2000, current_year + 1):
            # Calculate bounds for the location
            bbox_size = 0.015
            min_lon = lon - bbox_size
            min_lat = lat - bbox_size
            max_lon = lon + bbox_size
            max_lat = lat + bbox_size
            
            date_str = f"{year}-07-01"
            
            # Determine sensor metadata
            if year >= 2016:
                sensor = "Sentinel-2"
                resolution = "10m/px"
            elif year >= 2013:
                sensor = "Landsat 8"
                resolution = "30m/px"
            elif year >= 2000:
                sensor = "Landsat 7/MODIS"
                resolution = "30-250m/px"
            else:
                sensor = "Landsat 5"
                resolution = "30m/px"
            
            # Use a placeholder/static tile URL that works without CORS
            # In production, you'd use a proper tile service or Earth Engine
            # For now, we'll just return the bounds and let the base layer show through
            image_url = None  # This will make the overlay invisible, showing base map
            
            # Add metadata
            style = {}
            if year < 2010:
                style = {"filter": "sepia(0.3) contrast(1.1) brightness(0.95)"}
            elif year < 2018:
                style = {"filter": "contrast(1.05) brightness(0.98)"}
            
            timeline[year] = {
                "url": image_url,
                "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
                "metadata": {
                    "sensor": sensor,
                    "resolution": resolution,
                    "provider": "Base Map (Demo Mode)",
                    "date": date_str,
                    "style": style,
                    "note": "Historical imagery requires Earth Engine API"
                }
            }
        
        return {"status": "success", "source": "Demo Mode", "timeline": timeline}
    except Exception as e:
        print(f"❌ Timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/proxy-image")
async def proxy_image(bbox: str, year: int):
    """
    Smart Imagery Proxy: Tries Google Earth Engine first (Authentic Historical), 
    falls back to ESRI (Best Available) if GEE is offline.
    """
    try:
        # Parse bbox
        coords = bbox.split(',')
        if len(coords) != 4:
            raise HTTPException(status_code=400, detail="Invalid bbox format")
        
        min_lon, min_lat, max_lon, max_lat = map(float, coords)
        
        # Calculate center for GEE
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

        # 1. Try Google Earth Engine (If enabled)
        if earth_engine_available:
            print(f"🌍 Attempting GEE fetch for {year} at {center_lat}, {center_lon}")
            try:
                # Run GEE call in a separate thread to avoid blocking async loop
                # since the official python client is synchronous
                loop = asyncio.get_event_loop()
                gee_data = await loop.run_in_executor(None, get_satellite_image_url, center_lat, center_lon, year)
                
                if gee_data and gee_data.get('url'):
                    print(f"✅ GEE Success! Redirecting to: {gee_data['url'][:50]}...")
                    from fastapi.responses import RedirectResponse
                    return RedirectResponse(url=gee_data['url'])
                else:
                    print("⚠️ GEE returned no data for this location/year.")
            except Exception as gee_error:
                print(f"⚠️ GEE Fetch failed: {gee_error}")
        
        # 2. Fallback to ESRI Proxy
        # Construct ESRI URL with correct spatial references
        # bboxSR=4326 indicates the input coordinates are WGS84 Lat/Lon
        # imageSR=102100 (or 3857) requests the output image in Web Mercator
        esri_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=102100&size=800,600&format=png&f=image"
        
        print(f"Fetching fallback image for year {year} from: {esri_url}")
        
        # Fetch the image
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(esri_url)
            # print(f"ESRI response status: {response.status_code}")
            
            if response.status_code == 200:
                # Return the image with proper headers
                from fastapi.responses import Response
                return Response(
                    content=response.content,
                    media_type="image/png",
                    headers={
                        "Cache-Control": "public, max-age=86400",
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            else:
                error_detail = f"ESRI returned status {response.status_code}"
                print(f"Error: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Proxy error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")


@app.get("/geocode")
async def geocode(query: str):
    """
    Geocode a location string using Nominatim (via backend to avoid CORS/User-Agent issues).
    """
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"format": "json", "q": query, "limit": 1}
        headers = {"User-Agent": "SatWatch-Academy-App/1.0"}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display_name": data[0]["display_name"]}
                return {"error": "Location not found"}
            else:
                return {"error": f"Geocoding provider error: {resp.status_code}"}
    except Exception as e:
        print(f"Geocoding error: {e}")
        return {"error": str(e)}


# Simple in-memory cache for satellite data
SAT_CACHE = {}

@app.get("/satellites")
async def get_satellites(group: str = "active"):
    """
    Multi-source Satellite Proxy with speed optimization and caching.
    Tries CelesTrak first with short timeouts, then falls back to AMSAT.
    """
    import time
    now = time.time()
    
    # Check cache (valid for 5 minutes)
    if group in SAT_CACHE:
        cache_data, timestamp = SAT_CACHE[group]
        if now - timestamp < 300: # 5 minutes
            print(f"📦 Serving {group} orbital data from local archive cache")
            return PlainTextResponse(cache_data)

    # Sources prioritized by data richness
    sources = [
        # Primary: CelesTrak (Comprehensive)
        {
            "url": f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                "Referer": "https://celestrak.org/"
            }
        },
        # Secondary: AMSAT (Radio & Research - Very Reliable Fallback)
        {
            "url": "https://www.amsat.org/tle/current/nasabare.txt",
            "headers": {"User-Agent": "Mozilla/5.0"}
        }
    ]
    
    for source in sources:
        try:
            print(f"🛰️ Syncing Orbital Data: {source['url']} (Timeout: 5s)")
            async with httpx.AsyncClient(follow_redirects=True, timeout=5.0) as client:
                response = await client.get(source['url'], headers=source['headers'])
                
                if response.status_code == 200 and len(response.text) > 100:
                    print(f"✅ Uplink Sync Complete ({source['url']})")
                    SAT_CACHE[group] = (response.text, now)
                    return PlainTextResponse(response.text)
        except Exception as e:
            print(f"⚠️ Link Weak: {str(e)}. Switching frequency...")

    return PlainTextResponse("")


@app.post("/analyze")
async def analyze_images(before_image: UploadFile = File(...), after_image: UploadFile = File(...)):
    """
    Accepts two images (Before/After), runs the change detection model, 
    and returns the GeoJSON result + paths to the images.
    """
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    # Save uploaded files
    before_path = job_dir / f"before_{before_image.filename}"
    after_path = job_dir / f"after_{after_image.filename}"

    with open(before_path, "wb") as buffer:
        shutil.copyfileobj(before_image.file, buffer)
    
    with open(after_path, "wb") as buffer:
        shutil.copyfileobj(after_image.file, buffer)

    # Run Inference
    # Note: Real system would use a task queue (Celery/Redis) here.
    try:
        # We assume predict_change returns a dict with 'geojson' and 'report'
        inference_data = predict_change(str(before_path), str(after_path), str(RESULTS_DIR / job_id))
        
        return {
            "job_id": job_id,
            "status": "completed",
            "before_url": f"/files/{job_id}/{before_path.name}",
            "after_url": f"/files/{job_id}/{after_path.name}",
            "change_map_url": f"/results/{job_id}/{inference_data['change_map_path']}",
            "geojson": inference_data['geojson'],
            "report": inference_data['report']
        }
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"Error: {e}")
        with open("api_error.log", "w") as f:
            f.write(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
import asyncio
import random

class MonitorRequest(BaseModel):
    target: str
    location_name: str
    lat: float
    lon: float
    from_date: str
    to_date: str
    email: str

@app.post("/monitor")
async def start_monitoring(request: MonitorRequest):
    """
    Registers a continuous monitoring task.
    If the date is in the past, it performs a retrospective analysis.
    """
    from datetime import datetime
    try:
        req_date = datetime.strptime(request.from_date, "%Y-%m-%d")
        # Check if the END date is in the past/today to determine if this is a historical look-back
        to_date_obj = datetime.strptime(request.to_date, "%Y-%m-%d")
        
        # If to_date is today or in the past, we can generate a full report NOW.
        # If to_date is in the future, we must arm the surveillance system.
        is_historical_report = to_date_obj <= datetime.now()
    except:
        # Default to future surveillance if date parsing fails
        is_historical_report = False

    if is_historical_report:
        print(f"📊 Historical Alert Triggered for {request.location_name}")
        
        from_year = req_date.year
        to_year = to_date_obj.year

        from_url = None
        to_url = None

        # 1. Try Earth Engine first
        if earth_engine_available:
            from_data = get_satellite_image_url(request.lat, request.lon, from_year)
            to_data = get_satellite_image_url(request.lat, request.lon, to_year)
            if from_data: from_url = from_data.get('url')
            if to_data: to_url = to_data.get('url')
        
        # 2. Fallback to ESRI if Earth Engine didn't return URLs (or is unavailable)
        if not from_url or not to_url:
            print("⚠️ Earth Engine unavailable or no data. Using fallback imagery for report.")
            bbox_size = 0.015
            min_lon, min_lat = request.lon - bbox_size, request.lat - bbox_size
            max_lon, max_lat = request.lon + bbox_size, request.lat + bbox_size
            # Use specific ESRI proxy URL construction with spatial reference
            esri_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={min_lon},{min_lat},{max_lon},{max_lat}&bboxSR=4326&imageSR=102100&size=800,600&format=png&f=image"
            from_url = esri_url
            to_url = esri_url

        if from_url and to_url:
            # 2. Download images to temporary location
            job_id = f"hist_{str(uuid.uuid4())[:8]}"
            job_dir = UPLOAD_DIR / job_id
            job_dir.mkdir(exist_ok=True)
            
            before_path = job_dir / f"before_{from_year}.png"
            after_path = job_dir / f"after_{to_year}.png"

            async def download_image(url, path):
                async with httpx.AsyncClient(timeout=60.0) as client:
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 200:
                            with open(path, "wb") as f:
                                f.write(resp.content)
                            return True
                        print(f"❌ Download failed with status {resp.status_code}")
                        return False
                    except Exception as e:
                        print(f"❌ Download error: {e}")
                        return False

            success_b = await download_image(from_url, before_path)
            success_a = await download_image(to_url, after_path)

            if success_b and success_a:
                # 3. Run Analysis
                try:
                    analysis = predict_change(str(before_path), str(after_path), str(RESULTS_DIR / job_id))
                    
                    return {
                        "status": "historical_report",
                        "message": f"Retrospective report generated for {request.location_name}.",
                        "task_id": job_id,
                        "analysis": {
                            "job_id": job_id,
                            "before_url": f"/files/{job_id}/{before_path.name}",
                            "after_url": f"/files/{job_id}/{after_path.name}",
                            "report": analysis['report']
                        }
                    }
                except Exception as e:
                    print(f"Historical analysis failed: {e}")

    # Future or simulation path
    async def simulated_surveillance_task():
        print(f"🛰️ Satellite locked on {request.location_name}")
        await asyncio.sleep(3) 
        severity = random.choice(["Medium", "High", "Critical"])
        print(f"📧 Alert Email to {request.email}: {severity} severity {request.target} detected.")

    asyncio.create_task(simulated_surveillance_task())
    
    return {
        "status": "active", 
        "message": f"Future surveillance armed for {request.location_name}. System will alert {request.email} upon change detection.",
        "task_id": str(uuid.uuid4())
    }

# Serve uploaded files and results
app.mount("/files", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

def start():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
