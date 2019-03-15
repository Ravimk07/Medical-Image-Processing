# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:32:11 2019

@author: E75849
"""

import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def Database_gen(folder):
    database = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            database.append(img_rgb)
            labels.append(filename)
    return database,labels


database, labels  = Database_gen('E:/sample data/croped data/')
print("Total Images : {}".format(len(labels)))

imageDB = pd.DataFrame()
imageDB['image_matrix'] = database

fig=plt.figure(figsize=(8, 10), dpi= 80, edgecolor='k')
plt.imshow(imageDB['image_matrix'][0],'gray'),plt.title(labels[0])

def conti(img):
    imag = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(imag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((20,20),np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    contours,hierachy=cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

imageDB['contour'] = imageDB['image_matrix'].apply(conti)


imageDB['Labels'] = labels
imageDB.head()

def Cbir(DB_image_contour):
    return cv2.matchShapes(DB_image_contour,contour_query,1,0.0)

directory = "E:/sample data/Query image/" 
#searchImage= directory + "BRT_18_003NG_TN_105_1-1_2018-02-22 10_18_17.jpg"

querys, query_labels  = Database_gen(directory)
print("Total Images : {}".format(len(query_labels)))

queryDB = pd.DataFrame()
queryDB['image_matrix'] = querys

queryDB['contour'] = queryDB['image_matrix'].apply(conti)


queryDB['Labels'] = query_labels
queryDB.head()

for i in range(len(query_labels)):
    contour_query = queryDB['contour'][i]
    imageDB['similarity'] = imageDB['contour'].apply(Cbir)
    print(queryDB['Labels'][i])
    print(imageDB.nsmallest(5, 'similarity')['Labels'])






























