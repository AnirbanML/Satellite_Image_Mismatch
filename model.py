from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import os


def get_inception(inp, filt):
    #inp = Input((224,224,3))
    #1X1 conv
    conv1 = Conv2D(filters = filt/4,kernel_size = (1,1), padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)

    #1X1_3X3 conv
    conv2 = Conv2D(filters = filt/4,kernel_size = (1,1), padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(filters = filt/4,kernel_size = (3,3), padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)

    #1X1_5X5 conv
    conv3 = Conv2D(filters = filt/4,kernel_size = (1,1), padding = 'same', kernel_initializer = 'he_normal')(inp)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filters = filt/4,kernel_size = (5,5), padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)

    #3X3 maxpool 1X1 conv
    pool4 = tf.nn.max_pool2d(inp, ksize = (3,3), strides=1, padding='SAME')
    conv4 = Conv2D(filters = filt/4,kernel_size = (1,1), padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)

    concat = concatenate([conv1, conv2, conv3, conv4])
    return(concat)


def get_Model():
    inpt = Input((224, 224, 3))
    print(inpt.shape)
    conv1 = get_inception(inpt, 64)
    conv1 = get_inception(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = get_inception(pool1, 128)
    conv2 = get_inception(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    conv3 = get_inception(pool2, 256)
    conv3 = get_inception(conv3, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = get_inception(pool3, 512)
    conv4 = get_inception(conv4, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)

    conv5 = get_inception(pool4, 1024)
    conv5 = get_inception(conv5, 1024)

    up1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(conv5)
    up1 = concatenate([up1, conv4])
    conv6 = get_inception(up1, 512)
    conv6 = get_inception(conv6, 512)

    up2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv6)
    up2 = concatenate([up2, conv3])
    conv7 = get_inception(up2, 256)
    conv7 = get_inception(conv7, 256)

    up3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv7)
    up3 = concatenate([up3, conv2])
    conv8 = get_inception(up3, 128)
    conv8 = get_inception(conv8, 128)

    up4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv8)
    up4 = concatenate([up4, conv1])
    conv9 = get_inception(up4, 64)
    conv9 = get_inception(conv9, 64)

    conv = Conv2D(3, (1, 1), activation='softmax', padding='same', kernel_initializer='he_normal')(conv9)
    print(conv.shape)

    model = Model(inpt, conv)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

