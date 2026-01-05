import os
import math
import requests
import geopandas as gpd
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from shapely.geometry import box
import contextily as ctx

def set_modules_to_autoreload():
    # Source - https://stackoverflow.com/a
    # Posted by Taytay
    # Retrieved 2025-12-28, License - CC BY-SA 4.0

    from IPython.core.getipython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell

    ipython: InteractiveShell|None  = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("load_ext", "autoreload")
        ipython.run_line_magic("autoreload", "2")
    
    return

def setup_folders(folder_name, sub_folders=['osm', 'gtfs', 'population']):
    """
    Create a main folder and specified sub-folders if they do not already exist.

    Parameters:
    folder_name (str): The name of the main folder to create.
    sub_folders (list): A list of sub-folder names to create within the main folder.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_name, sub_folder)
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
    
    return

def generate_points_on_land(p1, p2, spacing = 100):
    #p1 and p2 are WGS-84 points defining the bounding box
    #spacing is the grid spacing in meters

    centroid = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    print("Centroid:", centroid)

    #find what UTM coordinate system the centroid is in
    centroid = gpd.GeoDataFrame(geometry=[gpd.points_from_xy([centroid[0]], [centroid[1]])[0]], crs="EPSG:4326")
    centroid_utm_crs = utm_crs = centroid.estimate_utm_crs()
    centroid_utm = centroid.to_crs(centroid_utm_crs)

    print("Estimated UTM CRS:", utm_crs)

    #convert points to UTM
    bounding_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([p1[0], p2[0]], [p1[1], p2[1]]), crs="EPSG:4326")
    bounding_points_gdf = bounding_points_gdf.to_crs(centroid_utm_crs)

    #create a new gdf, grid_gdf, in the utm coordinate system. Generate a grid of points
    box_x_num_points = math.ceil((bounding_points_gdf.geometry.x.max() - bounding_points_gdf.geometry.x.min()) / spacing)
    box_y_num_points = math.ceil((bounding_points_gdf.geometry.y.max() - bounding_points_gdf.geometry.y.min()) / spacing)

    print("Number of points in x direction:", box_x_num_points)
    print("Number of points in y direction:", box_y_num_points)

    top_left = (centroid_utm.geometry.x.values[0] - box_x_num_points * spacing / 2,
                centroid_utm.geometry.y.values[0] + box_y_num_points * spacing / 2)
    
    bottom_right = (centroid_utm.geometry.x.values[0] + box_x_num_points * spacing / 2,
                    centroid_utm.geometry.y.values[0] - box_y_num_points * spacing / 2)
    
    x_tics = np.linspace(top_left[0], bottom_right[0], box_x_num_points)
    y_tics = np.linspace(bottom_right[1], top_left[1], box_y_num_points)

    grid = np.meshgrid(x_tics, y_tics)

    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid[0].flatten(), grid[1].flatten()), crs=centroid_utm_crs)
    grid_points = grid_points.to_crs("EPSG:4326")
    
    grid_points['id'] = grid_points.index
    print("Generated {} grid points.".format(len(grid_points)))

    land = gpd.read_file("land-polygons-complete-4326/land_polygons.shp")

    points_extents = gpd.GeoDataFrame(
        geometry=[box(*grid_points.total_bounds)],
        crs=grid_points.crs
    )
    #clip land to the extent of the grid points
    land = gpd.clip(land, points_extents)

    grid_points = gpd.clip(grid_points, land)

    return(grid_points, land)

def clear_r5rpy_cache():
    import shutil
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "r5py"
    shutil.rmtree(cache_dir, ignore_errors=True)

    return

def download_transitland_feeds_for_area(p1, p2, date, sub_folder):
    #p1 and p2 are tuples representing (x, y) in WGS-84 coordinates defining the bounding box
    #date is a string in the format "YYYY-MM-DD"

    import requests
    import pandas as pd
    import os

    #clear the gtfs folder
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    else:
        for file in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    api_key = os.environ.get('TRANSITLAND_API_KEY')

    headers = {
        'Api-Key': api_key,
    }

    base_url = "https://transit.land/api/v2/rest"

    bbox = "{},{},{},{}".format(p1[0], p1[1], p2[0], p2[1])

    request = requests.get(
        f"{base_url}/feed_versions",
        headers=headers,
        params={
            "bbox": bbox,
            "limit": 100
        }
    )

    #create pandas dataframe from json
    df = pd.DataFrame.from_dict(request.json()['feed_versions'])

    for i, row in df.iterrows():
        df.at[i, 'onestop_id'] = row['feed']['onestop_id']

    print("{} unique feeds found in area.".format(len(df.onestop_id.unique())))
    
    for onestop_id in df.onestop_id.unique():
        print("Checking archive for {}".format(onestop_id))
        request = requests.get(
            f"{base_url}/feed_versions",
            headers=headers,
            params={
                "feed_onestop_id": onestop_id,
                "fetched_before": date + "T00:00:00Z",
                "limit": 100
            }
        )
        
        if request.json()['feed_versions'] == []:
            print("No feed versions found for {} before {}, skipping.".format(onestop_id, date))
            continue
        
        feed_df = pd.DataFrame.from_dict(request.json()['feed_versions'])
        #convert earliest_calendar_date and latest_calendar_date to datetime
        feed_df['earliest_calendar_date'] = pd.to_datetime(feed_df['earliest_calendar_date'])
        feed_df['latest_calendar_date'] = pd.to_datetime(feed_df['latest_calendar_date'])

        #filter by date
        feed_df = feed_df[(feed_df['earliest_calendar_date'] <= pd.to_datetime(date)) & (feed_df['latest_calendar_date'] >= pd.to_datetime(date))]

        if feed_df.empty:
            print("No feed versions found for {} covering {}, skipping.".format(onestop_id, date))
            continue

        first_row = feed_df.iloc[0]

        print("Download feed for agency {}. Date range: {} to {}".format(onestop_id, first_row['earliest_calendar_date'], first_row['latest_calendar_date']))

        #check if date is actually in the date range. iF not, there may be an issue with the API search criteria for the feed versions. Alternatively, the feed is very old.
        if not (first_row['earliest_calendar_date'] <= pd.to_datetime(date) <= first_row['latest_calendar_date']):
            print("IF THIS MESSAGE IS TRIGGERED, THERE MAY BE A BUG. Date {} not in range for feed {}, skipping.".format(date, onestop_id))
        
        #download the feed
        sha1 = first_row['sha1']
        request = requests.get(base_url + "/feed_versions/" + sha1 + "/download", headers=headers)

        with open(sub_folder + "/" + date + "_" + onestop_id + ".zip", 'wb') as f:
            f.write(request.content)
         
    return

#download_transitland_feeds_for_area(( -123.3656, 48.4284), ( -123.2500, 48.5000), "2026-01-03", "transit_data")

def download_topography(p1, p2, parent_folder):
    api_key = os.environ.get('OPEN_TOPOGRAPHY_API_KEY')

    base_url = "https://portal.opentopography.org/API"

    #search OT catalogue based on bounding box for datasets

    params = {
        'demtype': 'SRTMGL1',
        'west': min(p1[0], p2[0]),
        'south': min(p1[1], p2[1]),
        'east': max(p1[0], p2[0]),
        'north': max(p1[1], p2[1]),
        'API_Key': api_key

    }

    response = requests.get(f"{base_url}/globaldem", params=params)

    if response.status_code == 200:
        with open(parent_folder + "/topography/topography.tif", 'wb') as f:
            f.write(response.content)
        print("Topography data downloaded successfully.")
    else:
        print("Failed to download topography data. Status code:", response.status_code)
        print("Response:", response.text)

        return
    
    # Open the elevation raster (e.g., SRTMGL1) with rasterio.
    # rasterio reads the raster along with its CRS (coordinate reference system)
    # and ensures it can be interpreted correctly in Python. 
    # This helps r5py apply elevation costs without running into projection errors
    # (like missing Bursa-Wolf parameters or unknown EPSG codes) because
    # rasterio standardizes the CRS and allows transformations if needed.

    import rasterio
    from rasterio.crs import CRS

    src = parent_folder + "/topography/topography.tif"
    dst = parent_folder + "/topography/topography-wgs.tif"

    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
        profile["crs"] = CRS.from_epsg(4326)  # Set to WGS84

        with rasterio.open(dst, "w", **profile) as out:
            out.write(ds.read())

    return



def plot_interpolated_grid(x, y, z, ax=None, land = None, additional_gdf = None, title = "Interpolated Grid", sub_title = None, zero_centred=False, contours=None, cmap='plasma_r', method='linear', grid_size=100):
    """
    Plot an interpolated 2D grid from scattered x, y, z points using imshow.
    """

    #convert x, y data to epsg=3857
    raw_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs="EPSG:4326").to_crs("EPSG:3857")
    x, y = raw_points_gdf.geometry.x.values, raw_points_gdf.geometry.y.values

    land = land.to_crs("EPSG:3857")

    rbf = Rbf(x, y, z, function=method)

    x_tics, y_tics = np.linspace(x.min(), x.max(), grid_size), np.linspace(y.min(), y.max(), grid_size)
    x_grid, y_grid = np.meshgrid(x_tics, y_tics)
    z_grid = rbf(x_grid, y_grid)
    
    grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_grid.flatten(), y_grid.flatten()), crs="EPSG:3857")
    
    land['is_land'] = True
    #create a mask based on a column called 'is_land' in the land geodataframe
    grid_gdf = grid_gdf.sjoin(land, how='left', predicate='within')
    grid_gdf['mask'] = grid_gdf['is_land'].isnull()
    grid_gdf = grid_gdf[['geometry', 'mask']]

    masked_image = np.ma.masked_array(z_grid, mask=grid_gdf['mask'].values.reshape(x_grid.shape))

    #add map to background. Show simple streets map from OpenStreetMap
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    land_points = grid_gdf.loc[~grid_gdf['mask']]
    ax.set_xlim(land_points.geometry.x.min(), land_points.geometry.x.max())
    ax.set_ylim(land_points.geometry.y.min(), land_points.geometry.y.max())

    print("X limits:", ax.get_xlim())
    print("Y limits:", ax.get_ylim())

    #generate image mask based on land

    #add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs='EPSG:3857')

    img = ax.imshow(
        masked_image,
        extent=(x_tics.min(), x_tics.max(), y_tics.min(), y_tics.max()),
        origin='lower',
        cmap=cmap,
        vmin= -max(abs(z_grid.min()), abs(z_grid.max())) if zero_centred else z_grid.min(),
        vmax= max(abs(z_grid.min()), abs(z_grid.max())),
        alpha=0.4
    )

    #draw contours at the 60 minute mark
    contour = ax.contour(
        x_grid,
        y_grid,
        z_grid,
        levels=contours,
        colors='black',
        linewidths=1
    ) if contours is not None else None

    #label contours
    if contours is not None and contours != [0]:
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')

    #plot additional gdf if provided
    if additional_gdf is not None:
        additional_gdf = additional_gdf.to_crs("EPSG:3857")
        additional_gdf.plot(ax=ax, color='blue')

    #hide x and y tics
    ax.set_xticks([])
    ax.set_yticks([])

    #title
    ax.set_title(title, fontsize=16)

    #remove all padding, we'll control this from the figure level
    ax.margins(0)

    #add subtitle if provided, just below the title
    if sub_title is not None:
        ax.text(0.5, 1.015, sub_title, transform=ax.transAxes, fontsize=12, ha='center')
        #move the title up by 20
        ax.set_title("", fontsize=16)
        ax.set_title(title, fontsize=16, pad=23)

    fig.colorbar(img, ax=ax, label='Minutes', shrink=0.8)

    return fig, ax

def scatter_plot(x,y,z):
    plt.scatter(x, y, c=z, s=5, cmap='plasma_r', )

        #determine the aspect ratio of the figure based on p1, p2

    centroid = [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2]
    print("Centroid:", centroid)

    #find what UTM coordinate system the centroid is in
    centroid = gpd.GeoDataFrame(geometry=[gpd.points_from_xy([centroid[0]], [centroid[1]])[0]], crs="EPSG:4326")
    centroid_utm_crs = utm_crs = centroid.estimate_utm_crs()

    print("Estimated UTM CRS:", utm_crs)

    #convert points to UTM
    bounding_points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x.min(), x.max()], [y.min(), y.max()]), crs="EPSG:4326")
    bounding_points_gdf = bounding_points_gdf.to_crs(centroid_utm_crs)
    print(bounding_points_gdf)

    aspect_ratio = abs((bounding_points_gdf.geometry.y.max() - bounding_points_gdf.geometry.y.min()) / (bounding_points_gdf.geometry.x.max() - bounding_points_gdf.geometry.x.min()))
    print("Aspect ratio (height/width):", aspect_ratio)

    #add map to background. Show simple streets map from OpenStreetMap
    import contextily as ctx
    ax = plt.gca()
    
    ax.set_box_aspect(aspect=aspect_ratio)
    
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs='EPSG:4326')

    #add legend
    plt.colorbar(label='Minutes')

    plt.show()