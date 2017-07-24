import numpy as np
import argparse
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the image file")
# args = vars(ap.parse_args())

# 载入图片并将图片转换灰度图
image = cv2.imread('D:\\PycharmProjects\\NFCM\\pic.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 显示图片
# cv2.imshow("Image", gray)
# cv2.waitKey(0)

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)




# blur and threshold the image
# 使用内核对梯度进行平均模糊
# 将模糊后的图片进行二值化，梯度图中任何小于等于50的像素设为0（黑色），其余设置为白色
blurred = cv2.blur(gradient, (4, 4))
(_, thresh) = cv2.threshold(blurred, 55, 55, cv2.THRESH_BINARY)


# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)  # find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one


(im, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
