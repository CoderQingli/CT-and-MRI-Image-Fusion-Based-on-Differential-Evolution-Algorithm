#!/usr/bin/python
# -*- coding:utf8 -*-
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 对两个图片先进行小波分解，分别将低频和高频部分进行融合，最后进行小波重建
def wavelet(img1, img2, windowsize, wavelevel):
    wave1 = pywt.wavedec2(img1, 'db2', level=wavelevel)
    wave2 = pywt.wavedec2(img2, 'db2', level=wavelevel)
    # 使用pywavelet库中的wavedec2函数对图片进行小波分解，此函数返回一个list
    # [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]，分别是逼近、水平细节、垂直细节、对角线细节
    # assert len(wave1) == len(wave2)
    res = []
    for k in range(len(wave1)):
        # 处理低频分量
        if k == 0:
            lowWeight1, lowWeight2 = GetLowWeight(wave1[k], wave2[k])  # 获得权重值
            lowres = np.zeros(wave2[k].shape)
            row, col = wave1[k].shape
            # 用加权平均法对低频分量进行融合
            for i in range(row):
                for j in range(col):
                    lowres[i, j] = lowWeight1 * wave1[k][i, j] + lowWeight2 * wave2[k][i, j]
            res.append(lowres)
            continue
        # 处理高频分量
        highres = []
        for array1, array2 in zip(wave1[k], wave2[k]):
            a_row, a_col = array1.shape
            highFreq = np.zeros((a_row, a_col))
            var1 = GetVariance(array1, windowsize)  # 获取每幅图的方差矩阵
            var2 = GetVariance(array2, windowsize)
            # 根据区域方差最大准则进行融合
            for i in range(a_row):
                for j in range(a_col):
                    if var1[i, j] > var2[i, j]:
                        highFreq[i, j] = array1[i, j]
                    else:
                        highFreq[i, j] = array2[i, j]
            highres.append(highFreq)
        res.append(tuple(highres))
    return pywt.waverec2(res, 'db2')  # 利用waverec2()函数进行小波重建

# 对于低频分量，计算两图的权重比，作为融合时的权重
def GetLowWeight(img1, img2):
    mean1, var1 = cv2.meanStdDev(img1)
    mean2, var2 = cv2.meanStdDev(img2)
    weight1 = var1 / (var1 + var2)
    weight2 = var2 / (var1 + var2)
    return weight1, weight2

# 对于高频分量，获取其各自的方差矩阵
def GetVariance(array,windowsize):
    row, col = array.shape
    variance = np.zeros((row, col))
    # 对于每一个像素点，不去计算这个点对于全图的方差
    # 而是计算这个点对于其附近一个窗口的局部方差，这样可以减轻分块效应的影响
    for i in xrange(row):
        for j in xrange(col):
            if i - windowsize > 0:
                up = i - windowsize
            else:
                up = 0
            if i + windowsize < row:
                down = i + windowsize
            else:
                down = row
            if j - windowsize > 0:
                left = j - windowsize
            else:
                left = 0
            if j + windowsize < col:
                right = j + windowsize
            else:
                right = col
            window = array[up:down, left:right]
            mean, var = cv2.meanStdDev(window)
            variance[i, j] = var
    return variance

# 进行小波变换融合的主程序
def WaveletFusion(img1, img2, windowsize, wavelevel, name):
    I1 = Image.open(img1).convert('L')
    I2 = Image.open(img2).convert('L')
    I1 = I1.resize((I2.size), Image.ANTIALIAS)
    # 后续操作需要进行矩阵操作，故将灰度图转为矩阵格式
    I1array = np.array(I1)
    I2array = np.array(I2)
    res = wavelet(I1array, I2array, windowsize, wavelevel)
    Plot(I1, I2, res)
    picres = Image.fromarray(res).convert('RGB')
    picres.save(name)

#画图函数
def Plot(img1, img2, img):
    plt.subplot(131)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    CT = "CT-processed.jpg"
    T1 = "T1.jpg"
    T2 = "T2.jpg"
    WaveletFusion(CT, T1, 5, 2, "CT&T1-wavelet.jpg")
    WaveletFusion(CT, T2, 5, 2, "CT&T2-wavelet.jpg")