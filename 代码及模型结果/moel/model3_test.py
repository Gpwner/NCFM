# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:55:07 2017

@author: admin
"""

import numpy as np

np.random.seed(2016)
import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import KFold
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_opening, label
from scipy.spatial import ConvexHull
from skimage.filters import threshold_yen, threshold_triangle
from PIL import Image, ImageDraw

#==============================================================================
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
    #rtn = np.copy(mask)
    
    regions, n_labels = label(mask)
    label_list = range(1, n_labels+1)
    sizes = []
    for l in label_list:
        size = (regions==l).sum()
        sizes.append((size, l))

    sizes = sorted(sizes, reverse=True)
    labels = []
    #if sizes :                 
    #print(sizes)
    num_regions = min(num_regions, n_labels-1)
    #print(num_regions)
    min_size = sizes[num_regions][0]
    #print(min_size)
   

    
    for s, l in sizes:
        if s < min_size:
            regions[regions==l] = 0
        else:
            labels.append(l)

            
    #print(regions)
    #print(labels)
    #print("===============================")
    return regions, labels


def build_binary_opening_structure(binary_image, weight=1):
    s = 1 + 10000 * (binary_image.sum() / binary_image.size) ** 1.4
    s = int(max(12, 3 * np.log(s) * weight))
    return np.ones((s, s))


def simple_detector(o_image):
    
    
    image = np.asarray(o_image)
    
    dilation_iterations=40
    num_regions=4
    
    image_array = []
    titles = []

    image_array.append(image.astype('uint8'))
    titles.append('Original')

    # yen
    #threshold = threshold_yen(image)
    threshold = threshold_triangle(image)
    yen = np.zeros_like(image)
    yen[image[:,:,0] > threshold] = image[image[:,:,0] > threshold]

    # denoise
    binary_image = yen[:,:,0] > 0
    structure = build_binary_opening_structure(binary_image)
    binary_image = binary_opening(binary_image, structure=structure)
    binary_image = binary_dilation(binary_image, iterations=dilation_iterations)

    # mask
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
        return o_image
          
#==============================================================================


def load_image(filename):
    with open(filename, 'rb') as f:
            return np.asarray(Image.open(f))


def get_im_cv2(path):
    img = cv2.imread(path)
    new = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
    return new

def load_test():
    path = os.path.join('..', 'dataset1', 'test2', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        print(flbase)
        flbase = 'test_stg2/'+flbase
        img = load_img(os.path.join(fl))
        #img = get_im_cv2(fl)
        img = simple_detector(img)
        
        img = img_to_array(img)
        img = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
        
        X_test.append(img)
        X_test_id.append(flbase)

    return X_test, X_test_id


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)


def read_and_normalize_train_data():
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_process_test(models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'test1_sub.csv'
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    models = []
    model1 = load_model('F:\\code\\model_res\\model3-PIL2.0-CNN2.0\\Num_folds_3_nb_epoch_10\\1.h5')
    models.append(model1)
    model2 = load_model('F:\\code\\model_res\\model3-PIL2.0-CNN2.0\\Num_folds_3_nb_epoch_10\\2.h5')

    run_cross_validation_process_test(models)