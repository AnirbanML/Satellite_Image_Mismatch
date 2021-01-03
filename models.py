from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def jaccard_coef(y_true, y_pred):
    smooth = 1e-6
    intersection = K.sum(K.abs(y_true * y_pred), axis=[0, -1, -2])
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def get_unet():
    inp = Input((224, 224, 3))

    conv1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.5)(conv2)
    conv2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    conv3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.5)(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.5)(conv4)
    conv4 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)

    up1 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv5)
    up1 = concatenate([up1, conv4])
    conv6 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)

    up2 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv6)
    up2 = concatenate([up2, conv3])
    conv7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = BatchNormalization()(conv7)

    up3 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv7)
    up3 = concatenate([up3, conv2])
    conv8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Dropout(0.5)(conv8)
    conv8 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = BatchNormalization()(conv8)

    up4 = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(conv8)
    up4 = concatenate([up4, conv1])
    conv9 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.5)(conv9)
    conv9 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Activation('relu')(conv10)
    conv10 = Dropout(0.5)(conv10)
    conv10 = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Activation('relu')(conv10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Activation('sigmoid')(conv10)
    # print(inp.shape)
    # print(conv10.shape)

    model = Model(inp, conv10)

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, 'accuracy'])
    return model


def get_inception(inp, filt):
    # 1X1 conv
    conv1 = Conv2D(filters=int(filt / 4), kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inp)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)

    # 1X1_3X3 conv
    conv2 = Conv2D(filters=int(filt / 4), kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inp)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(filters=int(filt / 4), kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)

    # 1X1_5X5 conv
    conv3 = Conv2D(filters=int(filt / 4), kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(inp)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(filters=int(filt / 4), kernel_size=(5, 5), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)

    # 3X3 maxpool 1X1 conv
    pool4 = tf.nn.max_pool2d(inp, ksize=(3, 3), strides=1, padding='SAME')
    conv4 = Conv2D(filters=int(filt / 4), kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(pool4)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)

    concat = concatenate([conv1, conv2, conv3, conv4])
    # concat = BatchNormalization()(concat)
    return (concat)


def get_model_incept():
    inpt = Input((224, 224, 3))
    # print(inpt.shape)
    conv1 = get_inception(inpt, 16)
    conv1 = get_inception(conv1, 16)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    conv2 = get_inception(pool1, 32)
    conv2 = get_inception(conv2, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    conv3 = get_inception(pool2, 64)
    conv3 = get_inception(conv3, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv3)

    conv4 = get_inception(pool3, 128)
    conv4 = get_inception(conv4, 128)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)

    conv5 = get_inception(pool4, 256)
    conv5 = get_inception(conv5, 256)

    up1 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv5)
    up1 = concatenate([up1, conv4])
    conv6 = get_inception(up1, 128)
    conv6 = get_inception(conv6, 128)

    up2 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv6)
    up2 = concatenate([up2, conv3])
    conv7 = get_inception(up2, 64)
    conv7 = get_inception(conv7, 64)

    up3 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv7)
    up3 = concatenate([up3, conv2])
    conv8 = get_inception(up3, 32)
    conv8 = get_inception(conv8, 32)

    up4 = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(conv8)
    up4 = concatenate([up4, conv1])
    conv9 = get_inception(up4, 16)
    conv9 = get_inception(conv9, 16)
    conv9 = Dropout(0.5)(conv9)

    conv = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(conv9)
    conv = Activation('sigmoid')(conv)

    model = Model(inpt, conv)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, 'accuracy'])
    return model