import argparse
import os
import torch
import glob
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from model import ChangeDetectionModel
from preprocess import load_and_preprocess, coregister_images

def mask_to_polygons(mask, transform):
    """
    Converts a binary mask to GeoJSON polygons.
    """
    # mask is (H, W) boolean or 0/1 uint8
    mask = mask.astype(np.uint8)
    
    results = []
    # shapes() returns a generator of (geojson_geometry, value)
    for geom, val in shapes(mask, masking=mask, transform=transform):
        if val == 1:
            results.append(shape(geom))
    
    return results

def run_inference(before_dir, after_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Load Model
    model = ChangeDetectionModel().to(device)
    model.eval()
    
    # Support multiple extensions
    extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    before_files = []
    after_files = []
    
    for ext in extensions:
        before_files.extend(glob.glob(os.path.join(before_dir, ext)))
        after_files.extend(glob.glob(os.path.join(after_dir, ext)))
        
    before_files.sort()
    after_files.sort()
    
    for f1, f2 in zip(before_files, after_files):
        print(f"Processing pair: {f1} | {f2}")
        
        # Load & Preprocess
        img1, profile1 = load_and_preprocess(f1)
        img2, _ = load_and_preprocess(f2)
        
        # Co-register
        # In a real pipeline, check coverage/overlap first
        img2_aligned = coregister_images(img1, img2)
        
        # Prepare tensors
        t1 = torch.tensor(img1).unsqueeze(0).to(device)
        t2 = torch.tensor(img2_aligned).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output_mask = model(t1, t2)
            
        # Thresholding
        pred_mask = (output_mask.squeeze().cpu().numpy() > 0.5)
        
        # Post-process: Polygonize
        polygons = mask_to_polygons(pred_mask, profile1['transform'])
        
        if polygons:
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=profile1['crs'])
            out_name = os.path.basename(f1).replace('.tif', '_change.geojson')
            out_path = os.path.join(output_dir, out_name)
            gdf.to_file(out_path, driver='GeoJSON')
            print(f"Saved {len(polygons)} detection(s) to {out_path}")
        else:
            print("No changes detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--before_dir', required=True, help='Directory containing "before" images')
    parser.add_argument('--after_dir', required=True, help='Directory containing "after" images')
    parser.add_argument('--output_dir', required=True, help='Output directory for GeoJSONs')
    
    args = parser.parse_args()
    run_inference(args.before_dir, args.after_dir, args.output_dir)
