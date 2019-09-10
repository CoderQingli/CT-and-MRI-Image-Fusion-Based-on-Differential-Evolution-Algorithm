#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from math import *

def DE(img1, img2):
    #初始变量设定
    rows = len(img1)
    cols = len(img1[0])
    genes = rows * cols
    size = 20 # 每一代的个体数量
    times = 15 # 共进行times代操作
    low = np.array([0] * genes)
    high = np.array([255] * genes)
    f = 0.1 # 变异因子
    cr = 0.9 # 交叉概率因子
    tmp_res = 0

    #设置初始种群
    ori1 = img1.flatten()
    entropy1 = evaluate(ori1)
    print "Original entropy1 = ", entropy1 # 计算输入1的初始熵值
    ori2 = img2.flatten()
    entropy2 = evaluate(ori2)
    print "Original entropy2 = ", entropy2 # 计算输入2的初始熵值
    all = np.zeros((times, size, genes))
    for i in range(size/2):
        all[0][i] = ori1
        all[0][i+size/2] = ori2

    #进行差分进化算法部分
    for g in range(times-1):
        for indiv in range(size):
            # 从这一代种群中任选三个除当前个体外的个体进行变异操作
            g_without_indiv = np.delete(all[g], indiv, 0)
            np.random.shuffle(g_without_indiv)
            v_indiv = g_without_indiv[1] + f*(g_without_indiv[2] - g_without_indiv[3])
            # 处理边界越界问题
            v_indiv = [v_indiv[item] if v_indiv[item] < high[item] else high[item] for item in range(genes)]
            v_indiv = [v_indiv[item] if v_indiv[item] > low[item] else low[item] for item in range(genes)]
            # 进行交叉操作
            u_indiv = np.array([all[g][indiv][j] if (np.random.random() > cr) else v_indiv[j] for j in range(genes)])
            # 进行选择操作
            if evaluate(all[g][indiv]) < evaluate(u_indiv):
                all[g+1][indiv] = u_indiv
            else:
                all[g+1][indiv] = all[g][indiv]
        # 选择这一代中最优良的个体
        tmp_result = [evaluate(all[g+1][i]) for i in range(size)]
        if evaluate(all[g+1][np.argmax(tmp_result)]) > tmp_res:
            res = all[g+1][np.argmax(tmp_result)]
            tmp_res = evaluate(res)
    res = np.array(map(int, map(round, res)))
    print "Entropy after DE = ", tmp_res #计算结果的熵值
    return res.reshape(rows, cols)

def evaluate(Data):
    # 计算每个个体的熵值，熵值越大则包含的信息量越大
    Data = map(int, map(round, Data))
    num = len(Data)
    dic = {}
    for d in Data:
        if d not in dic:
            dic[d] = 1
        else:
            dic[d] += 1
    entro = 0
    for key in dic:
        p = float(dic[key])/num
        entro -= p * log(p, 2)
    return entro

def Plot(img1, img2, img):
    # 结果展示
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

def DEmain(img1, img2, name):
    I1 = Image.open(img1).convert('L')
    I2 = Image.open(img2).convert('L')
    I1 = I1.resize((I2.size), Image.ANTIALIAS)
    I1array = np.array(I1)
    I2array = np.array(I2)
    res = DE(I1array, I2array)
    Plot(I1, I2, res)
    scipy.misc.imsave(name, res) # 将所得结果进行保存

if __name__ == '__main__':
    i1 = "CT&T1-HSI.jpg"
    i2 = "CT&T1-wavelet.jpg"
    DEmain(i1, i2, "CT&T1-DE.jpg")