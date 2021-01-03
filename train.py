from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models import get_model_incept, get_unet
import os
import  tifffile as tiff
import numpy as np


trainX = []
trainY = []
for root, dirs, files in os.walk('./train'):
    for dir_name in dirs:
        image = tiff.imread('./train/'+str(dir_name)+'/'+str(dir_name)+'.tif')
        trainX.append(image/np.max(image))
        image = tiff.imread('./train/'+str(dir_name)+'/mask.tif')
        trainY.append(image/255)

model = get_unet() # or get_model_incept

checkpoint = [ModelCheckpoint(filepath='/content/checks',
                              save_weights_only=False,
                              monitor='val_accuracy',
                              mode='max',
                              save_best_only=True),
              ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1,
                                mode='min',
                                min_lr=0.00001,
                                min_delta=0.001,
                                verbose=1,
                                patience=10)]

valX = trainX[247:]
valY = trainY[247:]
val_data = (np.array(valX), np.array(valY))
results = model.fit(np.array(trainX[:247]), np.array(trainY[:247]),
                    validation_data=val_data,
                    batch_size=8,
                    epochs=250,
                    callbacks=checkpoint)