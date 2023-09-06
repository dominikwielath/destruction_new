import zarr
from pathlib import Path
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import shutil

## For local
# CITIES = ['aleppo', 'daraa']
#CITIES = ['moschun', 'volnovakha']
#OUTPUT_DIR = "../../test/mwd/outputs"
#DATA_DIR = "../../test/mwd/data"

## For artemisa
CITIES = ['hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakha', 'aleppo', 'damascus', 'daraa', 'deirezzor','hama', 'homs', 'idlib', 'raqqa']
OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs"
DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

## For workstation
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
#CITIES = ['aleppo', 'hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakha']
#CITIES = ['hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakha']
#OUTPUT_DIR = "/media/andre/Samsung8TB/mwd-latest/outputs"
#DATA_DIR = "/media/andre/Samsung8TB/mwd-latest/data"

TRAINING_DATA_DIR = OUTPUT_DIR + f"/data/{'-'.join(CITIES)}"

def read_zarr(city, suffix, path="../data"):
	path = f'{path}/{city}/others/{city}_{suffix}.zarr'
	return zarr.open(path)
	
im_te_pre = zarr.open(f"{TRAINING_DATA_DIR}/im_te_pre.zarr")
im_te_post = zarr.open(f"{TRAINING_DATA_DIR}/im_te_post.zarr")
la_te = zarr.open(f"{TRAINING_DATA_DIR}/la_te.zarr")

fp = open(f"{TRAINING_DATA_DIR}/metadata.txt")
for i, line in enumerate(fp):
	if i == 7:
		te_length = line.split("[")[-1].split("]")[0].replace("'", "").split(", ")
	if i > 10:
		break
fp.close()
te_length = [int(length) for length in te_length]

# Read per data per city
for i, city in enumerate(CITIES):
	im_te_pre_city = read_zarr(city, "im_te_pre", DATA_DIR)
	im_te_post_city = read_zarr(city, "im_te_post", DATA_DIR)
	la_te_city = read_zarr(city, "la_te", DATA_DIR)      

	if i == 0:
		city_im_pre = im_te_pre[:te_length[i],:,:,:]
		city_im_post = im_te_post[:te_length[i],:,:,:]
		city_la = la_te[:te_length[i]]
	else:
		previous_index_end = 0
		for j in range(i):
			previous_index_end += te_length[j]
		city_im_pre = im_te_pre[previous_index_end:previous_index_end+te_length[i],:,:,:]
		city_im_post = im_te_post[previous_index_end:previous_index_end+te_length[i],:,:,:]
		city_la = la_te[previous_index_end:previous_index_end+te_length[i]]
		sample_size = te_length[i]
		
			
	im_te_pre_equal = (im_te_pre_city[:] == city_im_pre[:]).all()
	im_te_post_equal = (im_te_post_city[:] == city_im_post[:]).all() 
	la_te_equal = (la_te_city[:] == city_la[:]).all()
	same_data = im_te_pre_equal & im_te_post_equal & la_te_equal
	
	im_te_pre_equal_shape = im_te_pre_city.shape == city_im_pre.shape
	im_te_post_equal_shape = im_te_post_city.shape == city_im_post.shape 
	la_te_equal_shape = la_te_city.shape == city_la.shape
	same_shape = im_te_pre_equal_shape & im_te_post_equal_shape & la_te_equal_shape
	
	print(f"For {city}, the split testset data match, shape match: {same_data}, {same_shape}")
		        
        
