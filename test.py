from model import get_Model
import tifffile as tiff
import cv2
import numpy as np


def get_predicton(data):
    model = get_Model()
    model.load_weights('incepunet_224X224X3_1000itr.h5')
    w, h = data.shape[:2]
    split = []
    hstart = 0
    for i in range(1, 9):
        hend = (h / 8) * i
        wstart = 0
        temp = []
        for j in range(1, 9):
            wend = (w / 8) * j
            test = data[int(hstart):int(hend), int(wstart):int(wend)]
            img = cv2.resize(test, (224, 224), cv2.INTER_AREA)
            img = img / np.max(img)
            # alpha = np.ones(img.shape[:2])
            # rgba = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])
            pred = model.predict(np.reshape(img, (-1, 224, 224, 3)))
            pred = pred * 255
            temp.append(pred)
            wstart = wend
        split.append(np.concatenate([temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]], axis=1))
        hstart = hend

    tiff.imshow(
        np.concatenate([split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7]], axis=0))

data = tiff.imread(r'./three_band/6120_2_2.tif').transpose([1, 2, 0])
get_predicton(data)