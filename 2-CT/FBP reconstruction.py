#!/usr/bin/python
# -*- coding:utf8 -*-

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fftshift,fft,ifft

theta_step=1
def pad_img(img):
  #进行补零操作，方便进一步处理
  s1,s2=img.size
  diag=int(np.ceil(np.sqrt(s1**2+s2**2)))
  img_pad=Image.new("L",(diag,diag))
  start_x=int(round((diag-s1)/2))
  start_y=int(round((diag-s2)/2))
  img_pad.paste(img,(start_x,start_y))
  return img_pad,start_x,start_y
  #记录原图粘贴的位置，方便全部处理完之后将补零的部分去掉

def projection(img,thetas):
  #进行投影操作
  num_theta=len(thetas)
  sinogram=np.zeros((img.size[1],num_theta))  #生成投影结果矩阵
  for i,theta in enumerate(thetas):
    rot_img=img.rotate(theta,resample=Image.BICUBIC)  #将图像旋转规定步长的角度
    sinogram[:,i]=np.sum(rot_img,axis=0)  #向x轴进行投影，由相加获得一个一维的数组
  return sinogram

def filter_projection(sinogram):
  #对获得的投影矩阵进行滤波处理
  #获得斜坡filter
  a=0.1
  size,num_thetas=sinogram.shape
  step=2*np.pi/size
  w=np.arange(-np.pi,np.pi,step)
  if len(w)<size:
    w=np.concatenate([w,w[-1]+step])
  rn1=np.abs(2/a*np.sin(a*w/2))
  rn2=np.sin(a*w/2)/(a*w/2)
  r=rn1*(rn2)**2
  #在频域将斜坡filter和投影矩阵进行相乘，再求ifft得到经过斜坡filter后的投影矩阵
  filter_=fftshift(r)
  filter_sinogram=np.zeros((size,num_thetas))
  for i in range(num_thetas):
    proj_fft=fft(sinogram[:,i])
    filter_proj=proj_fft*filter_
    filter_sinogram[:,i]=np.real(ifft(filter_proj))
  return filter_sinogram

# def back_projection(sinogram,thetas):
#   #一种求反投影的方法，但是效果不好
#   size_=sinogram.shape[0]
#   new_size=int(np.ceil(np.sqrt(size_**2+size_**2)))
#   start=int(round((new_size-size_)/2))
#   recon_img=Image.new("L",(new_size,new_size))
#
#   for i,theta in enumerate(thetas):
#     recon_img=recon_img.rotate(theta)
#     tmp1=sinogram[:,i]
#     tmp2=np.zeros(shape=(new_size))
#     tmp2[start:-start-1]=tmp1
#     tmp=np.repeat(np.expand_dims(tmp2,1),new_size,axis=1).T
#     recon_img+=tmp
#     recon_img=Image.fromarray(recon_img)
#     recon_img=recon_img.rotate(-theta)
#   recon_img=np.array(recon_img)
#   return recon_img[start:-start-1,start:-start-1]

def back_projection2(sinogram,thetas):
  #实际采用的求反投影的函数
  size_=sinogram.shape[0]
  recon_img=np.zeros((size_,size_))  #生成反投影矩阵

  for i,theta in enumerate(thetas):
    tmp1=sinogram[:,i]
    #取出投影矩阵第一列
    tmp=np.repeat(np.expand_dims(tmp1,1),size_,axis=1).T
    #将这一列进行复制得到一个和原图一样大小的矩阵，并进行转置
    #进行转置的原因是之前进行投影的时候，将投影到x轴的结果储存在了投影矩阵的一列中
    tmp=Image.fromarray(tmp)
    #将矩阵转换成一个图像
    tmp=tmp.rotate(theta,expand=0)
    #将图像旋转，和投影时候旋转的角度相同
    recon_img+=tmp
  return np.flipud(recon_img)

#主程序
img=Image.open("CT-processed.jpg").convert("L")
img_pad,start_x,start_y=pad_img(img)
thetas=np.arange(0,360,theta_step)
sinogram=projection(img_pad,thetas)
filter_sino=filter_projection(sinogram)

#直接对未过滤波器的投影矩阵进行BP重建
recon_img_bp=back_projection2(sinogram,thetas)
recon_img_bp=recon_img_bp[start_x:-start_x,start_y:-start_y]
recon_img_bp=np.round((recon_img_bp-np.min(recon_img_bp))/np.ptp(recon_img_bp)*255)
#对经过滤波器的投影矩阵进行FBP重建
recon_img_fbp=back_projection2(filter_sino,thetas)
recon_img_fbp=recon_img_fbp[start_x:-start_x,start_y:-start_y]
recon_img_fbp=np.round((recon_img_fbp-np.min(recon_img_fbp))/np.ptp(recon_img_fbp)*255)

#结果展示
fig1=plt.figure()
fig1.add_subplot(121)
plt.title("projection")
plt.imshow(sinogram,cmap=plt.get_cmap("gray"))
fig1.add_subplot(122)
plt.title("projection after filter")
plt.imshow(filter_sino,cmap=plt.get_cmap("gray"))
plt.show()

fig2=plt.figure()
fig2.add_subplot(121)
plt.title("bp reconstruction")
plt.imshow(recon_img_bp,cmap=plt.get_cmap("gray"))
fig2.add_subplot(122)
plt.title("fbp reconstruction")
plt.imshow(recon_img_fbp,cmap=plt.get_cmap("gray"))
plt.show()