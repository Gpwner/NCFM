
# coding: utf-8

# In[ ]:

import os
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential, load_model
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils

from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import operator

from subprocess import check_output

import sys
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, label
from scipy.spatial import ConvexHull
from skimage.filters import threshold_yen
from PIL import Image, ImageDraw

#########################################################################

# 目标检测
def mask_polygon(verts, shape):
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T


def convex_hull_mask(data, mask=True):
    segm = np.argwhere(data)
    hull = ConvexHull(segm)
    verts = [(segm[v,0], segm[v,1]) for v in hull.vertices]
    return mask_polygon(verts, data.shape)


def extract_largest_regions(mask, num_regions=2):

    regions, n_labels = label(mask)
    label_list = range(1, n_labels+1)
    sizes = []
    for l in label_list:
        size = (regions==l).sum()
        sizes.append((size, l))
        
    labels = []
    sizes = sorted(sizes, reverse=True)
    if sizes :
        print(sizes)
        num_regions = min(num_regions, n_labels-1)
        min_size = sizes[num_regions][0]

        
        for s, l in sizes:
            if s < min_size:
                regions[regions==l] = 0
            else:
                labels.append(l)
    else :
        print('--------------')

    return regions, labels


def build_binary_opening_structure(binary_image, weight=1):
    s = 1 + 10000 * (binary_image.sum()/binary_image.size) ** 1.4
    s = int(max(5, 3 * np.log(s) * weight))
    return np.ones((s, s))


def simple_detector(imageName):    
    dilation_iterations=40
    num_regions=2
    
    image = np.asarray(imageName)
    image_array = []
    titles = []

    image_array.append(image.astype('uint8'))
    titles.append('Original')

    
    # 图像自动阈值分割  http://www.cnblogs.com/denny402/p/5131004.html
    # threshold_otsu 基于Otsu的阈值分割方法
    
    # yen
    threshold = threshold_yen(image)   #返回一个阈值
    yen = np.zeros_like(image)
    yen[image[:,:,0] > threshold] = image[image[:,:,0] > threshold]

    # denoise  图像去噪
    binary_image = yen[:,:,0] > 0
    structure = build_binary_opening_structure(binary_image)
    binary_image = binary_opening(binary_image, structure=structure)
    binary_image = binary_dilation(binary_image, iterations=dilation_iterations)

    # mask
    # num_regions = 2
    regions, labels = extract_largest_regions(binary_image, num_regions=num_regions)
    
    
    if labels :
        mask = convex_hull_mask(regions>0)
        masked = np.zeros_like(image)
        masked[mask>0,:] = image[mask>0,:]

        titles.append('Whale')
        image_array.append(masked.astype('uint8'))
    
        myImage = Image.fromarray(np.uint8(image_array[1]))
        return myImage
    
    else :
        return imageName

#########################################################################

np.random.seed(7)
label_map = dict(zip([i for i in os.listdir('F:/code/data1/train') if not i=='.DS_Store'], range(8)))
model=load_model('F:\\code\\model_res\\model1-CNN1.0\\model1_epoch_10.h5')
test_im_names = []
test_imgs = []

for fname in os.listdir('F:/code/dataset1/test2'):
    if fname == '.DS_Store':
        continue
    test_im_names.append(fname)
    o_im = load_img(os.path.join('F:/code/dataset1/test2', fname))
    im=simple_detector(o_im)  
    im_data = img_to_array(im)
    im_data.resize(100, 100, 3)
    test_imgs.append(im_data)
    
print("Number of images", len(test_imgs))
print("Image size:", test_imgs[0].shape)

X_test_final = np.array(test_imgs)
X_test_final.astype('float32')
X_test_final /= 255.

final_preds = model.predict(X_test_final)

prediction_df = pd.DataFrame(final_preds)
prediction_df.columns = sorted(label_map.items(), key=operator.itemgetter(1))
prediction_df['image'] = test_im_names
print(prediction_df)
print("Done!")



# In[ ]:



