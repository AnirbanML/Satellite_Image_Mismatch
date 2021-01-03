import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import cv2
import concurrent.futures
from flask import Flask, request
import base64
from PIL import Image
import concurrent.futures
import time
import os
import json
from models import get_unet, get_model_incept
import requests


def get_prediction(data, model_name):

    w, h = data.shape[:2]
    split = []
    hstart = 0
    if ('model_trees' in model_name) or ('model_crops' in model_name) or ('model_road' in model_name):
        model = get_unet()
    else:
        model = get_model_incept()

    model.load_weights(model_name)
    for i in range(1, 9):
        hend = (h / 8) * i
        wstart = 0
        temp = []
        for j in range(1, 9):
            wend = (w / 8) * j
            test = data[int(hstart):int(hend), int(wstart):int(wend)]
            img = cv2.resize(test, (224, 224), cv2.INTER_AREA)
            img = img / np.max(img)
            predicted = model.predict(np.reshape(img, (-1, 224, 224, 3)))
            predicted = predicted[0]
            temp.append(np.round(predicted))
            wstart = wend
        split.append(np.concatenate([temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7]], axis=1))
        hstart = hend
    pred = np.concatenate([split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7]], axis=0)
    return pred


def getmask(img):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executer:
        models = ['/model_building.h5',
                  '/unet_model_road.h5',
                  '/unet_model_trees.h5',
                  '/unet_model_crops.h5']

        start = time.perf_counter()
        results = list(executer.map(get_prediction, [img] * len(models), models))

        pred_building = results[0]
        R_building = pred_building[:, :, 0] * 255
        G_building = pred_building[:, :, 1] * 70
        B_building = pred_building[:, :, 2] * 50

        pred_road = results[1]
        R_road = pred_road[:, :, 0] * 255
        G_road = pred_road[:, :, 1] * 242
        B_road = pred_road[:, :, 2] * 94

        pred_trees = results[2]
        R_trees = pred_trees[:, :, 0] * 94
        G_trees = pred_trees[:, :, 1] * 242
        B_trees = pred_trees[:, :, 2] * 94

        pred_crops = results[3]
        R_crops = pred_crops[:, :, 0] * 7
        G_crops = pred_crops[:, :, 1] * 124
        B_crops = pred_crops[:, :, 2] * 7

        R_buff1 = np.maximum(R_building, R_road)
        G_buff1 = np.maximum(G_building, G_road)
        B_buff1 = np.maximum(B_building, B_road)

        R_buff2 = np.maximum(R_crops, R_trees)
        G_buff2 = np.maximum(G_crops, G_trees)
        B_buff2 = np.maximum(B_crops, B_trees)

        R = np.maximum(R_buff1, R_buff2)
        G = np.maximum(G_buff1, G_buff2)
        B = np.maximum(B_buff1, B_buff2)

        rgb = cv2.merge([R, G, B])
        stop = time.perf_counter()
        print("Prediction time -> ", round(stop - start, 2))
        return (rgb)


app = Flask(__name__)
#run_with_ngrok(app)

def download_image(url):
    downurl = 'https://drive.google.com/uc?export=download&id=' + url
    response = requests.get(downurl, stream=True)
    if response.status_code == 200:
        img = np.asarray(Image.open(response.raw))
        return img


@app.route('/')
def home():
    return "I am Live"


@app.route('/predict', methods=['GET'])
def predict():
    start = time.perf_counter()
    url1 = request.args.get('imageone')
    # image1 = base64_to_array(temp)
    image1 = download_image(url1)
    url2 = request.args.get('imagetwo')
    # image2 = base64_to_array(temp)
    image2 = download_image(url2)

    mask1 = getmask(image1)
    mask2 = getmask(image2)

    try:
        os.makedirs('./mask')
    except FileExistsError:
        pass
    i = -1
    while (i < 0):
        i = -1
        if os.path.isdir('/mask'):
            print('mask folder created')
            plt.imsave('/mask/mask1.jpeg', np.round(mask1 / np.max(mask1), 2))
            plt.imsave('/mask/mask2.jpeg', np.round(mask2 / np.max(mask2), 2))
            break

    i = -1
    while (i < 0):
        i = -1
        if os.path.isfile('/mask/mask1.jpeg') and os.path.isfile('/mask/mask2.jpeg'):
            print('image saved')
            break

    with open('/mask/mask1.jpeg', 'rb') as data:
        mask1_b64 = base64.b64encode(data.read())
    with open('/mask/mask2.jpeg', 'rb') as data:
        mask2_b64 = base64.b64encode(data.read())

    stop = time.perf_counter()
    print("total time -> ", round(stop - start, 2))

    return json.dumps({'mask1': mask1_b64.decode('utf-8'), 'mask2': mask2_b64.decode('utf-8')})


if __name__ == '__main__':
    app.run()