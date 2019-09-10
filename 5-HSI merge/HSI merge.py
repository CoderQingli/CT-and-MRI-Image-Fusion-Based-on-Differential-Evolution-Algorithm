#!/usr/bin/python
# -*- coding:utf8 -*-
import cv2
import numpy as np
from math import *

def resizing(img):
    # 对图片进行大小调整，统一大小
    w = img.shape[0]
    h = img.shape[1]
    img = cv2.resize(img, (408, 432), interpolation=cv2.INTER_CUBIC)
    return(img)

def rgb2gray(img):
    # 将RGB图像转换为灰度图
    w = img.shape[0]
    h = img.shape[1]
    gray = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            gray[i,j] = 0.299*img[i,j,0] + 0.587*img[i,j,1] + 0.114*img[i,j,2]
    return(gray)

def rgb2hsi(img):
    # 将RGB图像转换为HSI图像
    w = img.shape[0]
    h = img.shape[1]
    b, g, r = cv2.split(img)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    res = img.copy()
    H, S, I = cv2.split(res)
    # 根据算法分类讨论
    for i in range(w):
        for j in range(h):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2 + (r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            if den == 0:
                    H = 0
            else:
                theta = float(np.arccos(num / den))
                if b[i, j] <= g[i, j]:
                    H = theta
                else:
                    H = 2*np.pi - theta
            min_RGB = min(b[i, j], g[i, j], r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum
            H = H/(2*np.pi)
            I = 0.299*r[i,j] + 0.587*g[i,j] + 0.114*b[i,j]
            res[i, j, 0] = H*255
            res[i, j, 1] = S*255
            res[i, j, 2] = I*255
    return res

def equal(image):
    # 对图片进行局部直方图均衡化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    res = clahe.apply(gray)
    return res

def ReplaceI(gray, HSI):
    # HSI图像分解后，用灰度图替换HSI图像中的I分量，再恢复成HSI图像
    H, S, I = cv2.split(HSI)
    r, c = I.shape
    for i in range(r):
        for j in range(c):
            I[i,j] = gray[i][j]
    res = cv2.merge([H, S, I])
    return res

def hsi2rgb(img):
    # 将HSI图像转换为RGB图像
    h = img.shape[0]
    w = img.shape[1]
    H, S, I = cv2.split(img)
    # 归一化到[0,1]
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    res = img.copy()
    B, G, R = cv2.split(res)
    # 根据算法分类讨论
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * cos(H[i, j]* pi/180)) / cos((60 - H[i, j])* pi/180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * cos(H[i, j]* pi/180)) / cos((60 - H[i, j])* pi/180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * cos(H[i, j]* pi/180)) / cos((60 - H[i, j])* pi/180))
                    R = 3 * I[i, j] - (G + B)
            res[i, j, 0] = B * 255
            res[i, j, 1] = G * 255
            res[i, j, 2] = R * 255
    return res

def HSImain():
    CT = cv2.imread("CT-processed.jpg")
    T1 = cv2.imread("T1.jpg")
    T2 = cv2.imread("T2.jpg")
    CTresize = resizing(CT)
    T1resize = resizing(T1)
    T2resize = resizing(T2)
    CTgray = equal(CTresize)
    CThsi = rgb2hsi(CTresize)
    T1gray = equal(T1resize)
    T1hsi = rgb2hsi(T1resize)
    T2gray = equal(T2resize)
    T2hsi = rgb2hsi(T2resize)

    mergeCTt1 = ReplaceI(CTgray, T1hsi)
    mergeCTt2 = ReplaceI(CTgray, T2hsi)
    merget1CT = ReplaceI(T1gray, CThsi)
    merget2CT = ReplaceI(T2gray, CThsi)
    res1 = hsi2rgb(mergeCTt1) # CT作为高空间分辨率图与MRI的T1相融合
    cv2.imwrite('CT&T1-HSI.jpg', res1)
    res2 = hsi2rgb(mergeCTt2) # CT作为高空间分辨率图与MRI的T2相融合
    cv2.imwrite('CT&T2-HSI.jpg', res2)
    res3 = hsi2rgb(merget1CT) # MRI的T1相作为高空间分辨率图与CT相融合
    cv2.imwrite('T1&CT-HSI.jpg', res3)
    res4 = hsi2rgb(merget2CT) # MRI的T2相作为高空间分辨率图与CT相融合
    cv2.imwrite('T2&CT-HSI.jpg', res4)

    cv2.imshow("CT", CTresize)
    cv2.imshow("MRI's T1", T1resize)
    cv2.imshow("MRI's T2", T2resize)
    cv2.imshow("CT as high resolusion pic merge with T1", res1)
    cv2.imshow("CT as high resolusion pic merge with T2", res2)
    cv2.imshow("T1 as high resolusion pic merge with CT", res3)
    cv2.imshow("T2 as high resolusion pic merge with CT", res4)
    cv2.waitKey(0)

if __name__ == '__main__':
    HSImain()