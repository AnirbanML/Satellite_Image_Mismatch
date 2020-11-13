from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from model import get_Model
import os
import  tifffile as tiff
import numpy as np


trainX = []
trainY = []
for root, dirs, files in os.walk('./train'):
    for dir_name in dirs:
        image = tiff.imread('./train/'+str(dir_name)+'/'+str(dir_name)+'.tif')
        trainX.append(image)
        image = tiff.imread('./train/'+str(dir_name)+'/mask.tif')
        trainY.append(image)

model = get_Model()
checkpoint = ModelCheckpoint(filepath='./model_checkpoint',
                            save_weights_only=False,
                            monitor='val_accuracy',
                            mode='max',
                            save_best_only=True)

results = model.fit(np.array(trainX), np.array(trainY), validation_split= 0.05, batch_size=8, epochs=1000, callbacks=[checkpoint])