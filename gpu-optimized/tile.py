import os
import rasterio
import numpy as np
import re
import time
import random
import zarr
import shutil
import matplotlib.pyplot as plt
import gc

CITY = 'aleppo'
DATA_DIR = "../data"
TILE_SIZE = (128,128)
PRE_IMAGE_INDEX=[0,1]
SUFFIX = "im"

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
    return files

def pattern(city:str='.*', type:str='.*', date:str='.*', ext:str='tif') -> str:
    '''Regular expressions for search_data'''
    return f'^.*{city}/.*/{type}_{date}\.{ext}$'

def read_raster(source:str, band:int=None, window=None, dtype:str='uint8', profile:bool=False) -> np.ndarray:
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


def tile_sequences(images:np.ndarray, tile_size:tuple=(128, 128)) -> np.ndarray:
    '''Converts images to sequences of tiles'''
    n_images, image_width, image_height, n_bands = images.shape
    tile_width, tile_height = tile_size
    assert image_width  % tile_width  == 0
    assert image_height % tile_height == 0
    n_tiles_width  = (image_width  // tile_width)
    n_tiles_height = (image_height // tile_height)
    sequence = images.reshape(n_images, n_tiles_width, tile_width, n_tiles_height, tile_height, n_bands)
    sequence = np.moveaxis(sequence.swapaxes(2, 3), 0, 2)
    sequence = sequence.reshape(-1, n_images, tile_width, tile_height, n_bands)
    return sequence

def sample_split(images:np.ndarray, samples:dict) -> list:
    '''Splits the data structure into multiple samples'''
    samples = [images[samples == value, ...] for value in np.unique(samples)]
    return samples



def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if not os.path.exists(path):
        zarr.save(path, data)
    else:
        za = zarr.open(path, mode='a')
        za.append(data)


def delete_zarr_if_exists(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)

f = open(f"{DATA_DIR}/{CITY}/others/metadata.txt", "a")

def print_w(text):
    f.write(f"{text}\n")
    print(text)

pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/pre')
post_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{CITY}/images/post')
labels  = search_data(pattern(city=CITY, type='label'), directory=DATA_DIR)
samples = read_raster(f'{DATA_DIR}/{CITY}/others/{CITY}_samples.tif')

image_dates = sorted([el.split("image_")[1].split('.tif')[0] for el in [*post_images]])
label_dates = sorted([el.split("label_")[1].split('.tif')[0] for el in labels])


remove = []
for la in label_dates:
    if la.replace("-", "_") not in image_dates:
        remove.append(label_dates.index(la))

_ = []
_labels = []
for i, dt in enumerate(label_dates):
    if i not in remove:
        _.append(dt)
        _labels.append(labels[i])


label_dates = sorted(_)
labels = sorted(_labels)


suffixes = ["im_tr_pre", "im_va_pre", "im_te_pre", "im_tr_post", "im_va_post", "im_te_post",  "la_tr",  "la_va",  "la_te"]
for s in suffixes:
    delete_zarr_if_exists(CITY, s, DATA_DIR)


tot_tr = tot_va =tot_te =tot_un = tot_na= tot_ti= 0
for j, pre_image in enumerate(pre_images):
    pre_image = read_raster(pre_images[j])
    pre_image = tile_sequences(np.array([pre_image]), TILE_SIZE)


    print_w(f"\tTile counts:")
    for i in range(len(post_images)):
        image_date = post_images[i].split("image_")[-1].split(".tif")[0].replace("_", "-")
        label = labels[i]
        label = read_raster(label, 1)
        label = np.squeeze(label.flatten())

        total_count = len(label)
        tot_ti += total_count
        unique, counts = np.unique(label, return_counts = True)
        uncertain_counts = dict(zip(unique, counts))
        uncertain = 0
        if 99 in uncertain_counts.keys():
            uncertain = uncertain_counts[99]


        unc = np.where(label == 99)
        label = np.delete(label, unc, 0)

        image = post_images[i]
        image = read_raster(image)
        a = image
        image = tile_sequences(np.array([image]))
        image = np.squeeze(image)
        image = np.delete(image, unc, 0)

        _pre_image = np.delete(pre_image, unc, 0)
        samples_min_unc = np.delete(samples.flatten(), unc)

        _, pre_image_tr, pre_image_va, pre_image_te = sample_split(_pre_image, samples_min_unc)

        na_count, tr_count, va_count, te_count = _.shape[0], pre_image_tr.shape[0], pre_image_va.shape[0], pre_image_te.shape[0]
        tot_tr += tr_count
        tot_va += va_count
        tot_te += te_count
        tot_un += uncertain
        tot_na += na_count
        print_w(f"\t\t{image_date}:\t\t\t Total: {total_count}, U: {uncertain}, NA: {na_count}, Tr: {tr_count}, Va: {va_count}, Te: {te_count} | Qualified tiles: {tr_count+va_count+te_count}")

        _, image_tr, image_va, image_te = sample_split(image, samples_min_unc) # for smaller samples there is no noanalysis class
        _, label_tr, label_va, label_te = sample_split(label, samples_min_unc)


        pre_image_tr = np.squeeze(pre_image_tr)

        save_zarr(pre_image_tr, CITY, 'im_tr_pre', path=DATA_DIR)
        save_zarr(image_tr, CITY, 'im_tr_post', path=DATA_DIR)
        save_zarr(label_tr, CITY, 'la_tr', path=DATA_DIR)

        pre_image_va = np.squeeze(pre_image_va)
        save_zarr(pre_image_va, CITY, 'im_va_pre', path=DATA_DIR)
        save_zarr(image_va, CITY, 'im_va_post', path=DATA_DIR)
        save_zarr(label_va, CITY, 'la_va', path=DATA_DIR)

        pre_image_te = np.squeeze(pre_image_te)
        save_zarr(pre_image_te, CITY, 'im_te_pre', path=DATA_DIR)
        save_zarr(image_te, CITY, 'im_te_post', path=DATA_DIR)
        save_zarr(label_te, CITY, 'la_te', path=DATA_DIR)


print_w(f"\tTotal images:\t\t\t\t {tot_ti}")
print_w(f"\t\tTotal tr images:\t\t {tot_tr} ({round((tot_tr/tot_ti)*100, 2)}%)")
print_w(f"\t\tTotal va images:\t\t {tot_va} ({round((tot_va/tot_ti)*100, 2)}%)")
print_w(f"\t\tTotal te images:\t\t {tot_te} ({round((tot_te/tot_ti)*100, 2)}%)")
print_w(f"\t\tTotal unc images:\t\t {tot_un} ({round((tot_un/tot_ti)*100, 2)}%)")
print_w(f"\t\tTotal na images:\t\t {tot_na} ({round((tot_na/tot_ti)*100, 2)}%)")
print_w(f"\t\tSum:\t\t\t\t {tot_na+tot_un+tot_te+tot_va+tot_tr}")


tr_pre = read_zarr(CITY, "im_tr_pre", DATA_DIR)
tr_post = read_zarr(CITY, "im_tr_post", DATA_DIR)
la_tr = read_zarr(CITY, "la_tr", DATA_DIR)
index = random.randint(0,tr_pre.shape[0] - 10)


fig, ax = plt.subplots(1,1,dpi=200)
ax.imshow(a)
plt.suptitle("Original Image")
plt.savefig(f"{DATA_DIR}/{CITY}/others/orig.png")
del a

fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(tr_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(tr_post[index:index+5]):
    ax[i+5].imshow(image)
for i, label in enumerate(la_tr[index:index+5]):
    ax[i].set_title(label==1)
plt.suptitle("Training set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/tr_samples.png")



te_pre = read_zarr(CITY, "im_te_pre", DATA_DIR)
te_post = read_zarr(CITY, "im_te_post", DATA_DIR)
la_te = read_zarr(CITY, "la_te", DATA_DIR)
index = random.randint(0,te_pre.shape[0] - 10)



fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(te_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(te_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Test set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/te_samples.png")

va_pre = read_zarr(CITY, "im_va_pre", DATA_DIR)
va_post = read_zarr(CITY, "im_va_post", DATA_DIR)
la_va = read_zarr(CITY, "la_va", DATA_DIR)
index = random.randint(0,va_pre.shape[0] - 10)


fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
ax = ax.flatten()
for i, image in enumerate(va_pre[index:index+5]):
    ax[i].imshow(image)
for i, image in enumerate(va_post[index:index+5]):
    ax[i+5].imshow(image)
plt.suptitle("Validation set (sample images; top=pre, bottom=post)")
plt.savefig(f"{DATA_DIR}/{CITY}/others/va_samples.png")

f.write(f"Training set: {tr_pre.shape[0]} observations\n")
f.write(f"Validation set: {va_pre.shape[0]} observations\n")
f.write(f"Test set: {te_pre.shape[0]} observations\n\n")
f.close()

