############################################################
# Python 3.5
# (C) DietCoke
# November 23, 2015
# License: MIT
############################################################

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_opening, label, binary_dilation
from scipy.spatial import ConvexHull
from skimage.filters import threshold_triangle


############################################################
# i/o

def subplot_images(images, titles=None, suptitle=False):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 12),
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable': 'box-forced'}
    )
    fig.suptitle(suptitle, fontsize=18)
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')

    plt.show()
    plt.close()


def load_image(filename):
    with open(filename, 'rb') as f:
        # 将 PIL Image 图片转换为 numpy 数组
        return np.asarray(Image.open(f))


############################################################
# geometry

def mask_polygon(verts, shape):
    # 将图片转换为灰度图
    img = Image.new('L', shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)
    return mask.T


def convex_hull_mask(data, mask=True):
    segm = np.argwhere(data)
    hull = ConvexHull(segm)
    verts = [(segm[v, 0], segm[v, 1]) for v in hull.vertices]
    return mask_polygon(verts, data.shape)


def extract_largest_regions(mask, num_regions=2):
    regions, n_labels = label(mask)  # n_labels=4
    label_list = range(1, n_labels + 1)
    sizes = []
    for l in label_list:
        size = (regions == l).sum()
        sizes.append((size, l))
    # 按照size的个数进行逆序排序
    sizes = sorted(sizes, reverse=True)
    num_regions = min(num_regions, n_labels - 1)
    min_size = sizes[num_regions][0]

    labels = []
    for s, l in sizes:
        if s < min_size:
            regions[regions == l] = 0
        else:
            labels.append(l)
    return regions, labels


def build_binary_opening_structure(binary_image, weight=1):
    s = 1 + 10000 * (binary_image.sum() / binary_image.size) ** 1.4
    s = int(max(12, 3 * np.log(s) * weight))
    return np.ones((s, s))


############################################################
# main

def simple_whale_detector(filename, dilation_iterations=40, num_regions=3):
    image = load_image(filename)
    image_array = []
    titles = []

    image_array.append(image.astype('uint8'))
    titles.append('Original')

    # 第二种图像自动阈值分割

    # threshold = threshold_yen(image)
    threshold = threshold_triangle(image)
    # threshold = threshold_li(image)
    # threshold = threshold_isodata(image)
    # 创建一个大小与image等大的数组
    yen = np.zeros_like(image)
    # 这样一行代码是将非目标区域的像素点置为0，至关重要
    yen[image[:, :, 0] > threshold] = image[image[:, :, 0] > threshold]
    # 降噪操作
    # 求出每一个像素点是否大于0,，1 or 0
    binary_image = yen[:, :, 0] > 0
    structure = build_binary_opening_structure(binary_image)
    # 先腐蚀再膨胀
    binary_image = binary_opening(binary_image, structure=structure)
    # 再做一次膨胀，扩充边缘或填充小的孔洞。
    binary_image = binary_dilation(binary_image, iterations=dilation_iterations)
    # mask
    regions, labels = extract_largest_regions(binary_image, num_regions=num_regions)
    mask = convex_hull_mask(regions > 0)
    masked = np.zeros_like(image)
    masked[mask > 0, :] = image[mask > 0, :]

    titles.append('fish')
    image_array.append(masked.astype('uint8'))

    # 保存图片
    # Image.fromarray(np.uint8(image_array[1])).save("pic2.jpg")
    # plot
    subplot_images(
        image_array,
        titles=titles,
        suptitle=filename.split('.')[0]
    )
if __name__ == '__main__':
    simple_whale_detector('D:\\PycharmProjects\\data\\test_stg1\\img_07833.jpg')