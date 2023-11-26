import zarr
from pathlib import Path
import os
import math
import numpy as np
from tensorflow.keras import backend, layers, models, callbacks, metrics
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
import random
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import time
import shutil

## For local
# CITIES = ['aleppo', 'daraa']
CITIES = ['moschun', 'volnovakha']
OUTPUT_DIR = "../../test/mwd/outputs"
DATA_DIR = "../../test/mwd/data"

## For artemisa
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
# OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs"
# DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

## For workstation
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
# CITIES = ['aleppo', 'hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakha']
# CITIES = ['hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakha']
# OUTPUT_DIR = "/media/andre/Samsung8TB/mwd-latest/outputs"
# DATA_DIR = "/media/andre/Samsung8TB/mwd-latest/data"


MODEL = "double"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cities", help="Cities, comma separated. Eg: aleppo,raqqa,damascus")
parser.add_argument("--model", help="One of snn, double")
parser.add_argument("--output_dir", help="Output dir")
parser.add_argument("--data_dir", help="Path to data dir")
parser.add_argument("--runs_dir", help="Name of the grid search dir")

parser.add_argument("--units", help="Units")
parser.add_argument("--dropout", help="Dropout")
parser.add_argument("--lr", help="Learning Rate")
parser.add_argument("--filters", help="Number of filters")
parser.add_argument("--batch_size", help="Batch Size")
parser.add_argument("--run_id", help="Run ID")

args = parser.parse_args()

print(args)

if args.cities:
    CITIES = [el.strip() for el in args.cities.split(",")]

if args.model:
    MODEL = args.model

if args.output_dir:
    OUTPUT_DIR = args.output_dir

if args.data_dir:
    DATA_DIR = args.data_dir

if args.runs_dir:
    RUNS_DIR = args.runs_dir 
    
    
def read_zarr(city, suffix, path="../data"):
    path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    return zarr.open(path)

def save_zarr(data, path):
    # path = f'{path}/{city}/others/{city}_{suffix}.zarr'

    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

def save_zarr_sfl(data, suffix, path="../data"):
    path = f'{path}/{suffix}.zarr'
    if not os.path.exists(path):
        zarr.save(path, data)        
    else:
        za = zarr.open(path, mode='a')
        za.append(data)

def delete_zarr_if_exists(path):
    # path = f'{path}/{city}/others/{city}_{suffix}.zarr'
    if os.path.exists(path):
        shutil.rmtree(path)

def make_tuple_pair(n, step_size):
    if step_size > n:
        return [(0,n)]
    iters = math.ceil(n/step_size*1.0)
    l = []
    for i in range(0, iters):
        if i == iters - 1:
            t = (i*step_size, n)
            l.append(t)
        else:
            t = (i*step_size, (i+1)*step_size)
            l.append(t)
    return l




Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#runs = [f for f in os.listdir(OUTPUT_DIR) if ".log" not in f]
#runs = [f for f in runs if ".DS_Store" not in f]
#run_id = len(runs)+1

if args.run_id:
    run_id = int(args.run_id)

print(f"\n\n### Run ID: {run_id} (use this code for dense_predict.py) \n\n")
time.sleep(5)

if args.runs_dir:
    RUN_DIR = OUTPUT_DIR + f"/runs/{RUNS_DIR}/{run_id}"
else:
    RUN_DIR = OUTPUT_DIR + f"/runs/{'-'.join(CITIES)}/{run_id}"
    
TRAINING_DATA_DIR = OUTPUT_DIR + f"/data/{'-'.join(CITIES)}"
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

f = open(f"{OUTPUT_DIR}/runs.log", "a")
f.write(f"Run {run_id}: {CITIES} \n")
f.close()

def shuffle(old="tr", new="tr_sfl", delete_old=False, block_size = 5000):
    images_pre = zarr.open(f'{TRAINING_DATA_DIR}/im_{old}_pre.zarr')
    images_post = zarr.open(f'{TRAINING_DATA_DIR}/im_{old}_post.zarr')
    labels = zarr.open(f'{TRAINING_DATA_DIR}/la_{old}.zarr')
    print("Shape before shuffle", images_pre.shape)
    print("Shape before shuffle", images_post.shape)

    n = images_pre.shape[0]
    blocks = make_tuple_pair(n, block_size)
    np.random.shuffle(blocks)

    for i, bl in enumerate(blocks):
        print(i+1, bl)
        im_pre = images_pre[bl[0]: bl[1]]
        im_post = images_post[bl[0]: bl[1]]
        la = labels[bl[0]: bl[1]]

        r = np.arange(0, bl[1] - bl[0])
        np.random.shuffle(r)
        # print(r)
        
        im_pre = im_pre[r]
        im_post = im_post[r]
        la = la[r]

        save_zarr_sfl(data=im_pre, suffix=f"im_{new}_pre", path=TRAINING_DATA_DIR)
        save_zarr_sfl(data=im_post, suffix=f"im_{new}_post", path=TRAINING_DATA_DIR)
        save_zarr_sfl(data=la, suffix=f"la_{new}", path=TRAINING_DATA_DIR)

    if delete_old == True:
        delete_zarr_if_exists(f"{TRAINING_DATA_DIR}/im_{old}_pre.zarr")
        delete_zarr_if_exists(f"{TRAINING_DATA_DIR}/im_{old}_post.zarr")
        delete_zarr_if_exists(f"{TRAINING_DATA_DIR}/la_{old}.zarr")


# im_tr_pre = None
# im_tr_post = None
# la_tr = None
# im_va_pre = None
# im_va_post = None
# la_va = None

im_va_pre_length = []
im_va_post_length = []
la_va_length = []

im_te_pre_length = []
im_te_post_length = []
la_te_length = []

if os.path.exists(TRAINING_DATA_DIR):
    print(f"Data already generated and available at: {TRAINING_DATA_DIR}")
    
    fp = open(f"{TRAINING_DATA_DIR}/metadata.txt")
    for i, line in enumerate(fp):
        if i == 4:
            va_length = line.split("[")[-1].split("]")[0].replace("'", "").split(", ")
        if i == 7:
            te_length = line.split("[")[-1].split("]")[0].replace("'", "").split(", ")
        if i > 10:
            break
    fp.close()
    
else:
    Path(TRAINING_DATA_DIR).mkdir(parents=True, exist_ok=True)      
    for city in CITIES:
        im_tr_pre = read_zarr(city, "im_tr_pre", DATA_DIR)
        im_tr_post = read_zarr(city, "im_tr_post", DATA_DIR)
        la_tr= read_zarr(city, "la_tr", DATA_DIR)

        im_va_pre = read_zarr(city, "im_va_pre", DATA_DIR)
        im_va_post = read_zarr(city, "im_va_post", DATA_DIR)
        la_va = read_zarr(city, "la_va", DATA_DIR)
	
        im_te_pre = read_zarr(city, "im_te_pre", DATA_DIR)
        im_te_post = read_zarr(city, "im_te_post", DATA_DIR)
        la_te = read_zarr(city, "la_te", DATA_DIR)

        im_va_pre_length.append(im_va_pre.shape[0])
        im_va_post_length.append(im_va_post.shape[0])
        la_va_length.append(la_va.shape[0])
     
        im_te_pre_length.append(im_te_pre.shape[0])
        im_te_post_length.append(im_te_post.shape[0])
        la_te_length.append(la_te.shape[0])
        
        print(f"{city}-tr_pre",im_tr_pre)
        
        steps = make_tuple_pair(im_tr_pre.shape[0], 100000) 
        for i, st in enumerate(steps):
            _im_tr_pre = im_tr_pre[st[0]:st[1]]
            _im_tr_post = im_tr_post[st[0]:st[1]]
            _la_tr = la_tr[st[0]:st[1]]

            save_zarr(_im_tr_pre, f"{TRAINING_DATA_DIR}/im_tr_pre.zarr")
            save_zarr(_im_tr_post, f"{TRAINING_DATA_DIR}/im_tr_post.zarr")
            save_zarr(_la_tr, f"{TRAINING_DATA_DIR}/la_tr.zarr")
            
            del _im_tr_pre, _im_tr_post, _la_tr
            print(f"{city} - TR: Copied {i+1} out of {len(steps)} blocks..")

        print(f"{city}-va_pre",im_va_pre)
        steps = make_tuple_pair(im_va_pre.shape[0], 50000) 
        for i, st in enumerate(steps):
            _im_va_pre = im_va_pre[st[0]:st[1]]
            _im_va_post = im_va_post[st[0]:st[1]]
            _la_va = la_va[st[0]:st[1]]

            save_zarr(_im_va_pre, f"{TRAINING_DATA_DIR}/im_va_pre.zarr")
            save_zarr(_im_va_post, f"{TRAINING_DATA_DIR}/im_va_post.zarr")
            save_zarr(_la_va, f"{TRAINING_DATA_DIR}/la_va.zarr")
            
            del _im_va_pre, _im_va_post, _la_va
            print(f"{city} - VA: Copied {i+1} out of {len(steps)} blocks..")

        print(f"{city}-te_pre",im_te_pre)
        steps = make_tuple_pair(im_te_pre.shape[0], 50000) 
        for i, st in enumerate(steps):
            _im_te_pre = im_te_pre[st[0]:st[1]]
            _im_te_post = im_te_post[st[0]:st[1]]
            _la_te = la_te[st[0]:st[1]]

            save_zarr(_im_te_pre, f"{TRAINING_DATA_DIR}/im_te_pre.zarr")
            save_zarr(_im_te_post, f"{TRAINING_DATA_DIR}/im_te_post.zarr")
            save_zarr(_la_te, f"{TRAINING_DATA_DIR}/la_te.zarr")

            del _im_te_pre, _im_te_post, _la_te
            print(f"{city} - TE: Copied {i+1} out of {len(steps)} blocks..")
    print('Shuffling step..')
    shuffle(old="tr", new="tr_sfl", delete_old=True, block_size=5000)
    shuffle(old="tr_sfl", new="tr", delete_old=True, block_size=5000*5)

    f = open(f"{TRAINING_DATA_DIR}/metadata.txt", "a")
    f.write(f"\n\n######## Cities: {CITIES} \n\n")
    f.write(f"Validation Set pre length: {im_va_pre_length} \n")
    f.write(f"Validation Set post length: {im_va_post_length} \n")
    f.write(f"Validation Set label length: {la_va_length} \n")
    f.write(f"Test Set pre length: {im_te_pre_length} \n")
    f.write(f"Test Set post length: {im_te_post_length} \n")
    f.write(f"Test Set label length: {la_te_length} \n")
    f.close()
    
    va_length = im_va_pre_length
    te_length = im_te_pre_length

va_length = [int(length) for length in va_length]
te_length = [int(length) for length in te_length]

def save_img(pre, post, labels, filename):
    random_index = random.randint(0,pre.shape[0] - 10)
    fig, ax = plt.subplots(2,5,dpi=200, figsize=(25,10))
    ax = ax.flatten()
    for i, image in enumerate(pre[random_index:random_index+5]):
        ax[i].imshow(image)
    for i, image in enumerate(post[random_index:random_index+5]):
        ax[i+5].imshow(image)
    for i, label in enumerate(labels[random_index:random_index+5]):
        ax[i].set_title(label==1)
    plt.suptitle("Pre-post")
    plt.savefig(f"{RUN_DIR}/{filename}")

im_tr_pre = zarr.open(f"{TRAINING_DATA_DIR}/im_tr_pre.zarr")
im_tr_post = zarr.open(f"{TRAINING_DATA_DIR}/im_tr_post.zarr")
la_tr= zarr.open(f"{TRAINING_DATA_DIR}/la_tr.zarr")

im_va_pre = zarr.open(f"{TRAINING_DATA_DIR}/im_va_pre.zarr")
im_va_post = zarr.open(f"{TRAINING_DATA_DIR}/im_va_post.zarr")
la_va = zarr.open(f"{TRAINING_DATA_DIR}/la_va.zarr")

im_te_pre = zarr.open(f"{TRAINING_DATA_DIR}/im_te_pre.zarr")
im_te_post = zarr.open(f"{TRAINING_DATA_DIR}/im_te_post.zarr")
la_te = zarr.open(f"{TRAINING_DATA_DIR}/la_te.zarr")

f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write(f"\n\n######## Run {run_id}: {CITIES} \n\n")
f.write(f"Training Set: {np.unique(la_tr[:], return_counts=True)} \n")
f.write(f"Validation Set: {np.unique(la_va[:], return_counts=True)} \n")
f.write(f"Test Set: {np.unique(la_te[:], return_counts=True)} \n")
f.close()



# Begin SNN Code

BATCH_SIZE = 32
PATCH_SIZE = (128,128)
FILTERS = [32]
DROPOUT = [0.12, 0.15]
EPOCHS = [70, 100]
UNITS = [64, 128]
LR = [0.00003, 0.0001]


if args.batch_size:
    BATCH_SIZE = int(args.batch_size)

def dense_block(inputs, units:int=1, dropout:float=0, name:str=''):
    tensor = layers.Dense(units=units, use_bias=False, kernel_initializer='he_normal', name=f'{name}_dense')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation')(tensor)
    tensor = layers.Dropout(rate=dropout, name=f'{name}_dropout')(tensor)
    return tensor 


def convolution_block(inputs, filters:int, dropout:float, name:str):
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution1')(inputs)
    tensor = layers.Activation('relu', name=f'{name}_activation1')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation1')(tensor)
    tensor = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', use_bias=False, kernel_initializer='he_normal', name=f'{name}_convolution2')(tensor)
    tensor = layers.Activation('relu', name=f'{name}_activation2')(tensor)
    tensor = layers.BatchNormalization(name=f'{name}_normalisation2')(tensor)
    tensor = layers.MaxPool2D(pool_size=(2, 2), name=f'{name}_pooling')(tensor)
    tensor = layers.SpatialDropout2D(rate=dropout, name=f'{name}_dropout')(tensor)
    return tensor

def distance_layer(inputs):
    input0, input1 = inputs
    distances = tf.math.reduce_sum(tf.math.square(input0 - input1), axis=1, keepdims=True)
    distances = tf.math.sqrt(tf.math.maximum(distances, tf.keras.backend.epsilon()))
    return distances



def encoder_block_separated(inputs, filters:int=1, dropout=0, n=1, name:str=''):	
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name=f'{name}_block1')	
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name=f'{name}_block2')	
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name=f'{name}_block3')	
    tensor  = convolution_block(tensor, filters=filters*8, dropout=dropout, name=f'{name}_block4')	
    tensor  = convolution_block(tensor, filters=filters*16, dropout=dropout, name=f'{name}_block5')	
    outputs = layers.GlobalAveragePooling2D(name=f'{name}_global_pooling')(tensor)	
    #outputs = layers.Flatten(name=f'{name}_flatten')(tensor)	
    return outputs

def encoder_block_shared(shape:tuple, filters:int=1, n=1, dropout=0):	
    inputs  = layers.Input(shape=shape, name='inputs')	
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name='block1')	
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name='block2')	
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name='block3')	
    tensor  = convolution_block(tensor, filters=filters*8, dropout=dropout, name='block4')	
    tensor  = convolution_block(tensor, filters=filters*16, dropout=dropout, name='block5')	
    outputs = layers.GlobalAveragePooling2D(name='global_pooling')(tensor)	
    encoder = models.Model(inputs=inputs, outputs=outputs, name='encoder')	
    return encoder




def siamese_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encoder_block = encoder_block_shared(shape=shape, **args_encode)
    encode1 = encoder_block(images1)
    encode2 = encoder_block(images2)
    # Hidden dense layers
    distance = distance_layer([encode1, encode2])
    # concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(distance, **args_dense, name='dense_block1')
    dense   = dense_block(dense,    **args_dense, name='dense_block2')
    dense   = dense_block(dense,    **args_dense, name='dense_block3')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='siamese_convolutional_network')
    return model

def difference_network(shape:tuple, args_encode:dict, args_dense:dict):
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')

    # Hidden convolutional layers (shared parameters)
    # tensor          = layers.Subtract(name="subtract")([images1, images2])
    
    encoder_block = encoder_block_shared(shape=shape, **args_encode)
    # encode1 = encoder_block(images1)
    tensor = encoder_block(images2)

    tensor          = dense_block(tensor, **args_dense, name='dense_block1')
    tensor          = dense_block(tensor, **args_dense, name='dense_block2')
    tensor          = dense_block(tensor, **args_dense, name='dense_block3')

    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(tensor)
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='diff')
    return model



def double_convolutional_network(shape:tuple, args_encode:dict, args_dense:dict):
    # Input layers
    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')
    # Hidden convolutional layers (shared parameters)
    encode1 = encoder_block_separated(images1, **args_encode, name='encoder1')
    encode2 = encoder_block_separated(images2, **args_encode, name='encoder2')
    # Hidden dense layers
    concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(concat, **args_dense, name='dense_block1')
    #dense   = dense_block(dense,  **args_dense, name='dense_block2')
    #dense   = dense_block(dense,  **args_dense, name='dense_block3')
    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='double_convolutional_network')
    return model

class SiameseGenerator(Sequence):
    def __init__(self, images, labels, batch_size=BATCH_SIZE, train=True):
        self.images_pre = images[0]
        self.images_post = images[1]
        self.labels = labels
        self.batch_size = batch_size
        self.train = train


        
        # self.tuple_pairs = make_tuple_pair(self.images_t0.shape[0], int(self.batch_size/4))
        # np.random.shuffle(self.tuple_pairs)
    def __len__(self):
        return len(self.images_pre)//self.batch_size    
    
    def __getitem__(self, index):
        X_pre = self.images_pre[index*self.batch_size:(index+1)*self.batch_size].astype('float') / 255.0
        X_post = self.images_post[index*self.batch_size:(index+1)*self.batch_size].astype('float') / 255.0
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]

        if self.train:
            return {'images_t0': X_pre, 'images_tt': X_post}, y
        else:
            return {'images_t0': X_pre, 'images_tt': X_post}
            


gen_tr = SiameseGenerator((im_tr_pre, im_tr_post), la_tr, batch_size=BATCH_SIZE)
gen_va = SiameseGenerator((im_va_pre, im_va_post), la_va, batch_size=BATCH_SIZE)

indices = np.random.randint(0, im_tr_pre.shape[0]//32, 5)

for j, ind in enumerate(indices):
    fig, ax = plt.subplots(2,8,dpi=400, figsize=(25,6))
    ax = ax.flatten()
    for i, image in enumerate(gen_tr.__getitem__(ind)[0]['images_t0'][0:8]):
        ax[i].imshow(image)
        ax[i].set_title(gen_tr.__getitem__(ind)[1][i] == 1)
    for i, image in enumerate(gen_tr.__getitem__(ind)[0]['images_tt'][0:8]):
        ax[i+8].imshow(image)
    plt.suptitle("Training set (sample images; top=pre, bottom=post)")
    plt.tight_layout()
    plt.savefig(f"{RUN_DIR}/traing_data_samples_{j+1}.png")



#----------------------------- Dominik 14.08.2023

class SubgroupValidationCallback(Callback):
    def __init__(self, cities, length_list, im_pre, im_post, la):
        super().__init__()
        self.cities = cities
        self.length_list = length_list
        self.im_pre = im_pre
        self.im_post = im_post
        self.la = la

    def calculate_metrics(self, city, gen_city, city_la):
        yhat_proba_city = self.model.predict(gen_city)
        yhat_proba_city = np.squeeze(yhat_proba_city)
        y_city = np.squeeze(city_la[0:(len(city_la)//BATCH_SIZE)*BATCH_SIZE])
        roc_auc_test_city = roc_auc_score(y_city, yhat_proba_city)
        return roc_auc_test_city

    def on_epoch_end(self, epoch, logs=None):
        # Calculate and print validation metrics for subgroups
        for i, city in enumerate(self.cities):
            if i == 0:
            	city_im_pre = self.im_pre[:self.length_list[i],:,:,:]
            	city_im_post = self.im_post[:self.length_list[i],:,:,:]
            	city_la = self.la[:self.length_list[i]]
            	print(0, self.length_list[i])
            else:
                previous_index_end = 0
                for j in range(i):
            	    previous_index_end += self.length_list[j]
                print(previous_index_end, previous_index_end+self.length_list[i])
                city_im_pre = self.im_pre[previous_index_end:previous_index_end+self.length_list[i],:,:,:]
                city_im_post = self.im_post[previous_index_end:previous_index_end+self.length_list[i],:,:,:]
                city_la = self.la[previous_index_end:previous_index_end+self.length_list[i]]
                        
            gen_city = SiameseGenerator((city_im_pre, city_im_post), city_la, batch_size=BATCH_SIZE)
            
            city_auc = self.calculate_metrics(city, gen_city, city_la)
            print(f"Epoch {epoch + 1} - Subgroup {city} - val_auc: {city_auc:.4f} \n")

# ------------------------------

print(im_va_pre.shape)

print("+++++++++", gen_tr.__len__())
MODEL_STORAGE_LOCATION = f"{RUN_DIR}/model"
Path(MODEL_STORAGE_LOCATION).mkdir(parents=True)
training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=2, restore_best_weights=True),
    callbacks.ModelCheckpoint(f'{MODEL_STORAGE_LOCATION}', monitor='val_auc', verbose=0, save_best_only=True, save_weights_only=False, mode='max')#,
#    SubgroupValidationCallback(CITIES, va_length, im_va_pre, im_va_post, la_va)
]


filters = random.choice(FILTERS)
dropout = random.choice(np.linspace(DROPOUT[0], DROPOUT[1]))
epochs = random.choice(np.arange(EPOCHS[0],EPOCHS[1]))
units = random.choice(UNITS)
lr = random.choice(LR)


if args.filters:
    filters = int(args.filters)

if args.units:
    units = int(args.units) 

if args.lr:
    lr = float(args.lr)

if args.dropout:
    dropout = float(args.dropout)

args  = dict(filters=filters, dropout=dropout, units=units) # ! Check parameters before run
args_dense  = dict(units=units, dropout=dropout)
parameters = f'batch_size={BATCH_SIZE} filters={filters}, dropout={np.round(dropout, 4)}, epochs={epochs}, units={units}, learning_rate={lr}'
print(parameters)
f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write(f"\n######## Run parameters \n\n{parameters}")
f.close()

if MODEL == 'snn':
    args_encode = dict(filters=filters, dropout=dropout, n_blocks=2, n_convs=2)
    model = siamese_convolutional_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense,
    )

if MODEL == 'double':
    args_encode = dict(filters=filters, dropout=dropout)
    model = double_convolutional_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense,
    )

if MODEL == 'diff':
    args_encode = dict(filters=filters, dropout=dropout,  n_blocks=5, n_convs=1)
    model = difference_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense,
    )

if MODEL == 'triple':
    args_encode = dict(filters=filters, dropout=dropout,  n_blocks=2, n_convs=3)
    model = double_convolutional_network(
        shape=(*PATCH_SIZE, 3),  
        args_encode = args_encode,
        args_dense = args_dense,
    )

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',metrics.AUC(num_thresholds=200, curve='ROC', name='auc')])
model.summary()

history = None

try:
  history = model.fit(
    gen_tr,
    validation_data=gen_va,
    epochs=epochs,
    verbose=1,
    callbacks=training_callbacks)
except:
  print("## Model training stopped, generating numbers on best model so far..")
  print("## Please wait, the program will terminate automatically..")
# Train model on dataset


def plot_training(H):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["accuracy"], label="train_accuracy")
	plt.plot(H.history["val_accuracy"], label="val_accuracy")
	plt.plot(H.history["auc"], label="train_auc")
	plt.plot(H.history["val_auc"], label="val_auc")
	plt.title(f"Training Accuracy and AUC")
	plt.suptitle(f"Cities = {CITIES}; RUN ID = {run_id}") 
	plt.xlabel("Epoch #")
	plt.ylabel("AUC")
	plt.text(0.65, 0.18, f"\nmax(val_auc)={np.round(np.max(H.history['val_auc']), 4)}", fontsize=8, transform=plt.gcf().transFigure)
	plt.legend(loc="lower left")
	plt.savefig(f"{RUN_DIR}/training.png")

if history:
    plot_training(history)

def calculate_metrics(best_model, city, gen_city, city_la):
    yhat_proba_city = best_model.predict(gen_city)
    yhat_proba_city = np.squeeze(yhat_proba_city)
    y_city = np.squeeze(city_la[0:(len(city_la)//BATCH_SIZE)*BATCH_SIZE])
    roc_auc_test_city = roc_auc_score(y_city, yhat_proba_city)
    return roc_auc_test_city

# model_path = f'{MODEL_DIR}/{CITY}/snn/run_{i}'
best_model = load_model(MODEL_STORAGE_LOCATION, custom_objects={'auc':metrics.AUC(num_thresholds=200, curve='ROC', name='auc')})
gen_te= SiameseGenerator((im_te_pre, im_te_post), la_te, train=False)
yhat_proba, y = np.squeeze(best_model.predict(gen_te)), np.squeeze(la_te[0:(len(la_te)//BATCH_SIZE)*BATCH_SIZE])
roc_auc_test = roc_auc_score(y, yhat_proba)
#calculate precision and recall
precision, recall, thresholds = precision_recall_curve(y, yhat_proba)


#create precision recall curve
fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
f = open(f"{RUN_DIR}/metadata.txt", "a")
f.write("\n\n######## Test set performance\n\n")
f.write(f'Test Set AUC Score for the ROC Curve: {roc_auc_test} \nAverage precision:  {np.mean(precision)}\n')
print(f"""
    Test Set AUC Score for the ROC Curve: {roc_auc_test} 
    Average precision:  {np.mean(precision)}
    Parameters: {parameters}
""")
# ------------------------------------ Dominik 14.08.2023
print(f"\n \n")
for i, city in enumerate(CITIES):
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
            gen_city = SiameseGenerator((city_im_pre, city_im_post), city_la, batch_size=BATCH_SIZE)
            city_auc = calculate_metrics(best_model, city, gen_city, city_la)
            print(f" - {city} - Sample size: {sample_size} - test_auc: {city_auc:.4f} \n")
            f.write(f" - {city} - Sample size: {sample_size} - test_auc: {city_auc:.4f} \n")
# --------
f.close()
#display plot
plt.savefig(f"{RUN_DIR}/pr_curve.png")

delete_zarr_if_exists(f"{RUN_DIR}/im_tr_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_tr_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_tr.zarr")

delete_zarr_if_exists(f"{RUN_DIR}/im_va_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_va_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_va.zarr")

delete_zarr_if_exists(f"{RUN_DIR}/im_te_pre.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/im_te_post.zarr")
delete_zarr_if_exists(f"{RUN_DIR}/la_te.zarr")





