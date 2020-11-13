# Dstl-Satellite-Imagery-Feature-Detection
Dstl provides you with 1km x 1km satellite images in both 3-band and 16-band formats. Your goal is to detect and classify the types of objects found in these regions. 
In this competition, Dstl provides you with 1km x 1km satellite images in both 3-band and 16-band formats. Your goal is to detect and classify the types of objects found in these regions. 
3- and 16-bands images
There are two types of imagery spectral content provided in this competition. The 3-band images are the traditional RGB natural color images.

Object types
In a satellite image, you will find lots of different objects like roads, buildings, vehicles, farms, trees, water ways, etc. Dstl has labeled 10 different classes:

1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
2. Misc. Manmade structures 
3. Road 
4. Track - poor/dirt/cart track, footpath/trail
5. Trees - woodland, hedgerows, groups of trees, standalone trees
6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
7. Waterway 
8. Standing water
9. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike

File descriptions
1. train_wkt.csv - the WKT format of all the training labels
    -ImageId - ID of the image
    -ClassType - type of objects (1-10)
    -MultipolygonWKT - the labeled area, which is multipolygon geometry represented in WKT format 
2. three_band.zip - the complete dataset of 3-band satellite images. The three bands are in the images with file name = {ImageId}.tif. MD5 = 7cf7bf17ba3fa3198a401ef67f4ef9b4 
3. sixteen_band.zip - the complete dataset of 16-band satellite images. The 16 bands are distributed in the images with file name = {ImageId}_{A/M/P}.tif. MD5 = e2949f19a0d1102827fce35117c5f08a
4. grid_sizes.csv - the sizes of grids for all the images
    -ImageId - ID of the image
    -Xmax - maximum X coordinate for the image
    -Ymin - minimum Y coordinate for the image
