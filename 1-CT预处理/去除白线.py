#!/usr/bin/python
# -*- coding:utf8 -*-
import cv2

def erase(startrow, startcol, img):
    if not(0 <= startrow < w and 0 <= startcol < h):
        return
    if img[startrow][startcol] == 0:
        return
    img[startrow][startcol] = 0
    erase(startrow-1, startcol, img)
    erase(startrow+1, startcol, img)
    erase(startrow, startcol-1, img)
    erase(startrow, startcol+1, img)

CT = cv2.imread("CT.jpg")
b, g, r = cv2.split(CT)
w, h = b.shape
erase(320, 2, b)
erase(377, 290, b)
erase(320, 2, g)
erase(377, 290, g)
erase(320, 2, r)
erase(377, 290, r)
res = cv2.merge([b, g, r])
cv2.imwrite('CT-processed.jpg',res)

cv2.imshow("res", res)
cv2.waitKey(0)