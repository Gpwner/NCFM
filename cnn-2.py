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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version

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


def extract_largest_regions(mask, num_regions=1):
    rtn = np.copy(mask)
    regions, n_labels = label(mask)
    label_list = range(1, n_labels+1)
    sizes = []
    for l in label_list:
        size = (regions==l).sum()
        sizes.append((size, l))

    sizes = sorted(sizes, reverse=True)
    num_regions = min(num_regions, n_labels-1)
    min_size = sizes[num_regions][0]

    labels = []
    for s, l in sizes:
        if s < min_size:
            regions[regions==l] = 0
        else:
            labels.append(l)

    return regions, labels


def build_binary_opening_structure(binary_image, weight=1):
    s = 1 + 10000 * (binary_image.sum()/binary_image.size) ** 1.4
    s = int(max(5, 3 * np.log(s) * weight))
    return np.ones((s, s))


def simple_whale_detector(imageName):    
    dilation_iterations=40
    num_regions=1
    
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
    mask = convex_hull_mask(regions>0)
    masked = np.zeros_like(image)
    masked[mask>0,:] = image[mask>0,:]  
    
    titles.append('Fish')
    image_array.append(masked.astype('uint8'))
      
    # save 
    myImage = Image.fromarray(np.uint8(image_array[1]))
    return myImage
#########################################################################



def get_im_cv2(path):
    img = cv2.imread(path)
    #new = cv2.resize(img, (16, 16), cv2.INTER_LINEAR)
    return img


def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        # 文件夹的下标
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        # 组合路径，，..\\NCFM\\data\\train\\fLd\\*.jpg
        path = os.path.join('..', 'dataset', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            ori_image = get_im_cv2(fl)
            img = simple_whale_detector(ori_image)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)  # what does this do?

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def load_test():
    path = os.path.join('..', 'dataset', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        #img = get_im_cv2(fl)
        ori_image = get_im_cv2(fl)
        img = simple_whale_detector(ori_image)
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


def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(2.5))  # changed this from 0.5
    model.add(Dense(8, activation='softmax'))
    # 梯度下降算法
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 32
    nb_epoch = 10
    random_state = 51

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0), ]
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score * len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score / len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
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
    info_string = 'loss_' + info_string \
                  + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    info_string, models = run_cross_validation_create_models(num_folds)
    run_cross_validation_process_test(info_string, models)
