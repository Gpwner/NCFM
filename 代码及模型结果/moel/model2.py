# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import
from __future__ import print_function

import os
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import operator

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



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
    mask = convex_hull_mask(regions>0)
    masked = np.zeros_like(image)
    masked[mask>0,:] = image[mask>0,:]  
    
    #titles.append('Fish')
    image_array.append(masked.astype('uint8'))
      
    # save 
    myImage = Image.fromarray(np.uint8(image_array[1]))
    return myImage
#########################################################################



np.random.seed(7)

labels = []
img_data = []

train_set_path = 'F:/code/data1/train'

label_map = dict(zip([i for i in os.listdir(train_set_path) if not i=='.DS_Store'], range(8)))

for d in os.listdir(train_set_path):
    print("Processing {}...".format(d))
    if d == '.DS_Store':
        continue
    for f in os.listdir(os.path.join(train_set_path, d)):
        labels.append(label_map[d])
        im = load_img(os.path.join(train_set_path, d, f))
        im = simple_detector(im)       
        im_data = img_to_array(im)
        im_data.resize(100, 100, 3)
        img_data.append(im_data)
print("Unique number of labels:", len(set(labels)))
print("Number of images:", len(img_data))
print("Length of label map:", len(label_map))


def cnn():
    model = Sequential()
    model.add(Convolution2D(128,4,4, border_mode='valid', input_shape=(100,100,3), activation='relu'))
    model.add(Convolution2D(128,4,4))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,3,3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(8))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X = np.array(img_data)
y = np.array(labels)

X = X.astype('float32')
y = y.astype('float32')
print("Data converted...")

X /= 255.

# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
y = np_utils.to_categorical(y)

model = cnn()
print("Model loaded...")
# model.fit(X_train, y_train, nb_epoch=10, batch_size=64, verbose=2)
result = model.fit(X, y, nb_epoch=10, batch_size=64, class_weight=None, 
                   verbose=1, shuffle=True,validation_split=0.2 )

result2 = model.evalute(X, y, batch_size=64, verbose=1, sample_weight=None)
#result = model.fit(data, label, batch_size=50,nb_epoch=35,shuffle=True,
#   verbose=1,show_accuracy=True,validation_split=0.2)

model.save('F:/model2_epoch_10.h5')

# preds = model.predict(X_test)
# print ("Log loss on hold-out set:", log_loss(y_test, preds))

plt.figure
plt.plot(result.epoch,result.history['acc'],label="acc")
plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
plt.scatter(result.epoch,result.history['acc'],marker='*')
plt.scatter(result.epoch,result.history['val_acc'])
plt.legend(loc='under right')
plt.show()

plt.figure
plt.plot(result.epoch,result.history['loss'],label="loss")
plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
plt.scatter(result.epoch,result.history['loss'],marker='*')
plt.scatter(result.epoch,result.history['val_loss'],marker='*')
plt.legend(loc='upper right')
plt.show()







#test_im_names = []
#test_imgs = []
#test_path = 'F:/code/dataset/test1'

#for fname in os.listdir(test_path):
#    if fname == '.DS_Store':
#        continue
#    test_im_names.append(fname)
#    o_im = load_img(os.path.join(test_path, fname))
#    im=simple_detector(o_im)  
#    im_data = img_to_array(im)
#    im_data.resize(100, 100, 3)
#    test_imgs.append(im_data)
#    
#print("Number of images", len(test_imgs))
#print("Image size:", test_imgs[0].shape)


#X_test_final = np.array(test_imgs)
#X_test_final.astype('float32')
#X_test_final /= 255.

#final_preds = model.predict(X_test_final)

#prediction_df = pd.DataFrame(final_preds)
#prediction_df.columns = sorted(label_map.items(), key=operator.itemgetter(1))
#prediction_df['image'] = test_im_names
#prediction_df.to_csv('submission.csv', encoding='utf8', index=False)




print("Done!")

