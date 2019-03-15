# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:56:09 2019

@author: E75849
"""

import numpy as np
import cv2
import os
from matplotlib import  pyplot as plt


imgs_path = "E:/sample data/"
imgs_dir = imgs_path + "BRT_18_003NG_TN_101_1-1_2018-02-22 10_14_36.scn_2.5ROI.jpg"
tem_dir = imgs_path + "1_template.png"

image = cv2.imread(imgs_dir)
template = cv2.imread(tem_dir)

image = cv2.resize(image,(1040,1040))
plt.imshow(image)

template = cv2.resize(template,(50,100))
plt.imshow(template)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.imshow(imageGray)

#template = cv2.Canny(template, 50, 200) 
plt.imshow(template)


result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_SQDIFF)
min_val1, max_val1, min_loc, max_loc = cv2.minMaxLoc(result)
plt.imshow(result)

va11 = min_val1;

#cv2.imshow('Detected',image)
 
#
#w, h = template.shape[:2]
#threshold = 0.001*min_val1
#loc = np.where( result >= threshold)
#
#for pt in zip(*loc[::-1]):
#    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#
#cv2.imwrite(imgs_path +"res.png", image)


imgs_path = "E:/sample data/"
imgs_dir = imgs_path + "BRT_18_003NG_TN_101_1-2_2018-02-22 10_23_22.scn_2.5ROI.jpg"
tem_dir = imgs_path + "1_template.png"

image2 = cv2.imread(imgs_dir)
template = cv2.imread(tem_dir)

image2 = cv2.resize(image2,(1040,1040))
plt.imshow(image2)

template = cv2.resize(template,(50,100))
plt.imshow(template)

imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
plt.imshow(imageGray)

#template = cv2.Canny(template, 50, 200) 
plt.imshow(template)


result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_SQDIFF)
min_val2, max_val2, min_loc, max_loc = cv2.minMaxLoc(result)
plt.imshow(result)

va12 = min_val2;






imgs_path = "E:/sample data/"
imgs_dir = imgs_path + "BRT_18_003NG_TN_102_1-1_2018-02-22 10_16_15.scn_2.5ROI.jpg"
tem_dir = imgs_path + "1_template.png"

image3 = cv2.imread(imgs_dir)
template = cv2.imread(tem_dir)

image3 = cv2.resize(image3,(1040,1040))
plt.imshow(image3)




