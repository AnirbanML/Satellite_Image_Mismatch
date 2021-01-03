import os
import csv
import sys
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import shapely
import numpy as np
import tifffile as tiff
import pandas as pd

#Create image mask
def mask_image(IM_ID):
 x_max = y_min = None
 for _im_id, _x, _y in csv.reader(open('./repo/grid_sizes.csv')):
  if _im_id == IM_ID:
   x_max, y_min = float(_x), float(_y)
   break

 maxInt = sys.maxsize
 while True:
  # decrease the maxInt value by factor 10
  # as long as the OverflowError occurs.

  try:
   csv.field_size_limit(maxInt)
   break
  except OverflowError:
   maxInt = int(maxInt / 10)

 # Load train poly with shapely
 component = []
 polyline = []
 for _im_id, _poly_type, _poly in csv.reader(open('./repo/train_wkt_v4.csv')):
  if _im_id == IM_ID:
   if _poly != 'MULTIPOLYGON EMPTY':
    component.append(_poly_type)
    polyline.append(shapely.wkt.loads(_poly))

 # Read image with tiff
 im_rgb = tiff.imread('./three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
 im_size = im_rgb.shape[:2]

 # Scale polygons to match image:
 h, w = im_size  # they are flipped so that mask_for_polygons works correctly
 w_ = w * (w / (w + 1))
 h_ = h * (h / (h + 1))
 x_scaler = w_ / x_max
 y_scaler = h_ / y_min

 train_polygons_scaled = []
 for poly in polyline:
  train_polygons_scaled.append(shapely.affinity.scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)))
 train_polygons = pd.DataFrame({'component': component, 'polyline': train_polygons_scaled})
 R = np.zeros(im_size, np.uint8)
 G = np.zeros(im_size, np.uint8)
 B = np.zeros(im_size, np.uint8)
 for m in range(train_polygons.shape[0]):

  '''Change values in if condition to select the following:
     1> Buildings - large building, residential, non-residential, fuel storage facility, fortified building
     2> Misc. Manmade structures 
     3> Road 
     4> Track - poor/dirt/cart track, footpath/trail
     5> Trees - woodland, hedgerows, groups of trees, standalone trees
     6> Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
     7> Waterway 
     8> Standing water
     9> Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle 
     10> Vehicle Small - small vehicle (car, van), motorbike'''

  if (train_polygons.values[m, 0] == '1') or (train_polygons.values[m, 0] == '2'):
   img_mask = np.zeros(im_size, np.uint8)
   int_coords = lambda x: np.array(x).round().astype(np.int32)
   exteriors = [int_coords(poly.exterior.coords) for poly in train_polygons.values[m, 1]]
   interiors = [int_coords(pi.coords) for poly in train_polygons.values[m, 1]
                for pi in poly.interiors]
   cv2.fillPoly(img_mask, exteriors, 1)
   cv2.fillPoly(img_mask, interiors, 0)
   R = np.maximum(R, 255 * img_mask)
   G = np.maximum(G, 255 * img_mask)
   B = np.maximum(B, 255 * img_mask)

 return cv2.merge([R, G, B])

def scale_percentile(matrix):
 w, h, d = matrix.shape
 matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
 # Get 2nd and 98th percentile
 mins = np.percentile(matrix, 1, axis=0)
 maxs = np.percentile(matrix, 99, axis=0) - mins
 matrix = (matrix - mins[None, :]) / maxs[None, :]
 matrix = np.reshape(matrix, [w, h, d])
 matrix = matrix.clip(0, 1)
 return matrix

#Divide image into dimentions 244X244
def crop_image(IM_ID):
 raw_img = tiff.imread('./three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
 mask = mask_image(IM_ID)
 w, h = raw_img.shape[:2]
 k = 1
 hstart = 0
 hend = 0
 i = 1

 while (hend < h):
  hend = 224 * i
  i = i + 1
  wstart = 0
  wend = 0
  j = 1
  while (wend < w) and (hend < h):
   wend = 224 * j
   j = j + 1

   FOLDER_NAME = IMG_NAME = IM_ID + '_' + str(k)
   k = k + 1
   os.makedirs('./train/{}'.format(FOLDER_NAME))

   img = raw_img[int(hstart):int(hend), int(wstart):int(wend)]
   tiff.imsave('./train/{}/{}.tif'.format(FOLDER_NAME, IMG_NAME), img)

   image = mask[int(hstart):int(hend), int(wstart):int(wend)]
   tiff.imsave('./train/{}/{}.tif'.format(FOLDER_NAME, 'mask'), image)
   wstart = wend
  hstart = hend

if __name__ == "__main__":
 train_wkt = pd.read_csv('./train_wkt_v4.csv')
 img_ids = train_wkt['ImageId'].unique()

 for img_name in img_ids:
  crop_image(img_name)