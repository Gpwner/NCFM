import cv2


path='F:/Kaggle_NCFM/dataset/train/train/ALB/img_00010.jpg'   
img = cv2.imread(path)   
cv2.namedWindow("Image")   
cv2.imshow("Image", img)
cv2.waitKey(0)   



