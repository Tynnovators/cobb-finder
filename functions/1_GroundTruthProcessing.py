import numpy as np
import cv2
from matplotlib import pyplot as  plt
img = cv2.imread(r'../download.png', cv2.IMREAD_GRAYSCALE)
#cv2.imwrite('C:/Users/Prathmesh/Documents/MATLAB/download.png', img)
img = cv2.resize(img, (600, 600))
cv2.imshow('b/w Image', img)
img = ~img

#img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray_filtered = cv2.bilateralFilter(img, 7, 50, 50)

edges_filtered = cv2.Canny(gray_filtered, 60, 120)

img_gray = ~edges_filtered

ret, thresh = cv2.threshold(img_gray, 10, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contours.pop(0)
#print(contours[0])
#print(len(contours))
cv2.drawContours(img_gray, contours, 6, (0, 55, 0), 3)
print(contours[6])
cv2.imshow('img',img)

#cv2.imshow('img_gray', img_gray)
#cv2.imshow('outline_vc.png', img_gray)
#gray_filtered = cv2.bilateralFilter(img, 7, 50, 50)

# Applying the canny filter
edges = cv2.Canny(img, 60, 120)
#edges_filtered = cv2.Canny(gray_filtered, 60, 120)
cv2.imshow('edges',edges)
#cv2.imshow('edges_filtered',edges_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()