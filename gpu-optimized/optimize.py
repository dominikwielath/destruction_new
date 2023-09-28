import tensorflow as tf
from tensorflow.keras import layers, models, metrics, callbacks

training_callbacks = [
    callbacks.EarlyStopping(monitor='val_auc', patience=2, restore_best_weights=True),
]


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

def encoder_block_separated(inputs, filters:int=1, dropout=0, n=1, name:str=''):	
    tensor  = convolution_block(inputs, filters=filters*1, dropout=dropout, name=f'{name}_block1')	
    tensor  = convolution_block(tensor, filters=filters*2, dropout=dropout, name=f'{name}_block2')	
    tensor  = convolution_block(tensor, filters=filters*4, dropout=dropout, name=f'{name}_block3')	
    tensor  = convolution_block(tensor, filters=filters*8, dropout=dropout, name=f'{name}_block4')	
    tensor  = convolution_block(tensor, filters=filters*16, dropout=dropout, name=f'{name}_block5')	
    outputs = layers.Flatten(name=f'{name}_flatten')(tensor)	
    return outputs

def create_model(trial):

    filters = trial.suggest_int("filters", [8,16,32])
    units = trial.suggest_int("units", [64, 128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.10, 0.20)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)


    args_encode = dict(filters=filters, dropout=dropout)
    args_dense  = dict(units=units, dropout=dropout)


    images1 = layers.Input(shape=shape, name='images_t0')
    images2 = layers.Input(shape=shape, name='images_tt')

     # Hidden convolutional layers (shared parameters)
    encode1 = encoder_block_separated(images1, **args_encode, name='encoder1')
    encode2 = encoder_block_separated(images2, **args_encode, name='encoder2')

    # Hidden dense layers
    concat  = layers.Concatenate(name='concatenate')(inputs=[encode1, encode2])
    dense   = dense_block(concat, **args_dense, name='dense_block1')
    dense   = dense_block(dense,  **args_dense, name='dense_block2')
    dense   = dense_block(dense,  **args_dense, name='dense_block3')

    # Output layer
    outputs = layers.Dense(units=1, activation='sigmoid', name='outputs')(dense)
    # Model
    model   = models.Model(inputs=[images1, images2], outputs=outputs, name='double_convolutional_network')

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    auc =  metrics.AUC(num_thresholds=200, curve='ROC', name='auc')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy',auc])


    return model

def ready_data():
    gen_tr = SiameseGenerator((im_tr_pre, im_tr_post), la_tr, batch_size=BATCH_SIZE)
    gen_va = SiameseGenerator((im_va_pre, im_va_post), la_va, batch_size=BATCH_SIZE)

    return gen_tr, gen_va

# Objective function
def objective(trial):
    # instantiate model
    model_opt = create_model(trial)


    training_data, validation_data = ready_data()

    # fit the model
    model_opt.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        verbose=1,
        callbacks=training_callbacks
    )
    
    # calculate accuracy score
    acc_score = model_opt.evaluate(x_test, y_test, verbose=0)[1]
    
    return acc_score