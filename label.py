import numpy as np
import rasterio
from rasterio import features
import os
import re
import geopandas
import time
from datetime import datetime
import pandas as pd
from collections import OrderedDict

pd.options.mode.chained_assignment = None  # default='warn'

CITY = "raqqa"
DATA_DIR = "../data"
ZERO_DAMAGE_BEFORE_YEAR = 2012
# PRE_IMAGE_INDEX = [0,1]
TILE_SIZE = (128,128)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--city", help="City")
parser.add_argument("--data_dir", help="Data Dir")
args = parser.parse_args()

if args.city:
    CITY = args.city

if args.data_dir:
    DATA_DIR = args.data_dir

def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    # if len(files) == 1: files = files[0]
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

def read_raster(source:str, band:int=None, window=None, dtype:str='int', profile:bool=False) -> np.ndarray:
    '''Reads a raster as a numpy array'''
    raster = rasterio.open(source)
    if band is not None:
        image = raster.read(band, window=window)
        image = np.expand_dims(image, 0)
    else: 
        image = raster.read(window=window)
    image = image.transpose([1, 2, 0]).astype(dtype)
    if profile:
        return image, raster.profile
    else:
        return image

def extract(files:list, pattern:str='\d{4}-\d{2}-\d{2}') -> list:
    pattern = re.compile(pattern)
    match   = [pattern.search(file).group() for file in files]
    return match

def rasterise(source, profile, attribute:str=None, dtype:str='float32') -> np.ndarray:
    '''Tranforms vector data into raster'''
    if isinstance(source, str): 
        source = geopandas.read_file(source)
    if isinstance(profile, str): 
        profile = rasterio.open(profile).profile
    geometries = source['geometry']
    if attribute is not None:
        geometries = zip(geometries, source[attribute])
    image = features.rasterize(geometries, out_shape=(profile['height'], profile['width']), transform=profile['transform'])
    image = image.astype(dtype)
    return image

def write_raster(array:np.ndarray, profile, destination:str, nodata:int=None, dtype:str='float64') -> None:
    '''Writes a numpy array as a raster'''
    if array.ndim == 2:
        array = np.expand_dims(array, 2)
    array = array.transpose([2, 0, 1]).astype(dtype)
    bands, height, width = array.shape
    if isinstance(profile, str):
        profile = rasterio.open(profile).profile
    profile.update(driver='GTiff', dtype=dtype, count=bands, nodata=nodata)
    with rasterio.open(fp=destination, mode='w', **profile) as raster:
        raster.write(array)
        raster.close()

def tiled_profile(source:str, tile_size:tuple=(128,128,1)) -> dict:
    '''Computes raster profile for tiles'''
    raster  = rasterio.open(source)
    profile = raster.profile
    assert profile['width']  % tile_size[0] == 0, 'Invalid dimensions'
    assert profile['height'] % tile_size[1] == 0, 'Invalid dimensions'
    affine  = profile['transform']
    affine  = rasterio.Affine(affine[0] * tile_size[0], affine[1], affine[2], affine[3], affine[4] * tile_size[1], affine[5])
    profile.update(width=profile['width'] // tile_size[0], height=profile['height'] // tile_size[0], count=tile_size[2], transform=affine)
    return profile


f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

def print_w(text):
    f.write(f"{text}\n")
    print(text)

image      = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)[0]
profile    = tiled_profile(image, tile_size=(*TILE_SIZE, 1))

# Reads damage reports
damage = search_data(f'{CITY}_damage.*gpkg$', directory=DATA_DIR)[0]
damage = geopandas.read_file(damage)
last_annotation_date = sorted(damage.columns)[-2]

# Extract report dates
dates = search_data(pattern(city=CITY, type='image'), directory=DATA_DIR)
dates = extract(dates, '\d{4}_\d{2}_\d{2}')
dates = list(map(lambda x: x.replace("_", "-"), dates))

# add additional date columns
known_dates = sorted(damage.drop('geometry', axis =1).columns)
known_dates.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'))

print_w(f"\tDates with annotations: \t\t {known_dates}")
print_w(f"\tMost recent date with annotations: \t {[last_annotation_date]}")

damage[list(set(dates) - set(damage.columns))] = np.nan
damage_columns = list(sorted(damage.columns))[:-1]
damage_columns.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'))
damage_columns.append("geometry")
damage = damage.reindex(damage_columns, axis=1)

# Set pre cols to 0
pre_cols = []
pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/pre')
pre_cols = [f.split("image_")[1].split(".tif")[0].replace("_", "-") for f in pre_images]
damage[pre_cols] = 0.0



f.write("\n\n######## Labeling Step\n\n")
# f.write(f"Using {pre_cols} as pre-dates\n")

post_cols = sorted([col for col in damage.drop('geometry', axis=1).columns if col not in pre_cols])
post_cols.sort(key=lambda date: datetime.strptime(date, '%Y-%m-%d'))


index_last_annotation_date = list(damage.columns).index(last_annotation_date)
columns_after_last_annotation_date = damage.columns[index_last_annotation_date+1:]
if len(columns_after_last_annotation_date != 1):
    columns_after_last_annotation_date = list(columns_after_last_annotation_date[:-1])
columns_until_last_annotation_date = damage.columns[0: index_last_annotation_date]
columns_until_last_annotation_date = list(damage.columns[0: index_last_annotation_date+1])

print_w(f"\tColumns until last annotation date: \t {columns_until_last_annotation_date}")
print_w(f"\tColumns after last annotation date: \t {columns_after_last_annotation_date}")


damage_full = damage.copy()
damage = damage_full[[*columns_until_last_annotation_date, 'geometry']]


# Corrected label assignment
geom = damage['geometry']
print_w(f"\tTotal coordinates: \t\t\t {[len(damage)]}")
damage = damage.drop('geometry', axis=1).T
for col in damage.columns:
    unc = np.where(damage[col].fillna(method='ffill') != damage[col].fillna(method='bfill'))
    if int(col)%3000 == 0 and int(col) !=0:
        print_w(f"\t\tProcessing annotations: \t {col} annotations processed")
    for i in unc:
        damage[col][i] = -1

for col in damage.columns:
    damage[col] = damage[col].fillna(method='ffill')

damage = damage.T
damage['geometry'] = geom
damage = damage

damage_after = damage_full[[last_annotation_date, *columns_after_last_annotation_date, 'geometry']]
for col in columns_after_last_annotation_date:
    damage_after[col] = np.where(damage_after[last_annotation_date] == 0.0, -1, damage_after[last_annotation_date]) * 1.0

damage.is_copy = None
damage_after.is_copy = None
final = pd.concat([damage,damage_after[columns_after_last_annotation_date]], axis=1)

# Writes damage labels
print_w(f"\tDatewise breakdown:")

for date in final.drop('geometry', axis=1).columns:
    subset = final[[date, 'geometry']].sort_values(by=date) # Sorting takes the max per pixel
    counts = subset[date].value_counts().to_dict()
    counts = dict(OrderedDict(sorted(counts.items())))
    print_w(f"\t\t{date}{' (gt):' if date in known_dates else ':    '} \t\t {counts}")
    subset[date] = np.where((subset[date] == -1.0) , 99.0, subset[date])
    subset[date] = np.where((subset[date] < 3), 0.0, subset[date])
    subset[date] = np.where((subset[date] == 3), 1.0, subset[date])
    subset = rasterise(subset, profile, date)
    write_raster(subset, profile, f'{DATA_DIR}/{CITY}/labels/label_{date}.tif', dtype='int8')
del date, subset

f.close()

