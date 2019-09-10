from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def pad_img(rows, cols, img):
  s1,s2=img.size
  img_pad=Image.new("L",(rows,cols))
  start_x=int(round((rows-s1)/2))
  start_y=int(round((cols-s2)/2))
  img_pad.paste(img,(start_x,start_y))
  return img_pad

def overlap(img1, img2, w, h):
  cnt = 0
  for i in range(h):
    for j in range(w):
      if img1[i][j] != 0 and img2[i][j] != 0:
        cnt += 1
  return cnt

def resizing(img, k):
  res = img.resize((int(CTw*k), int(CTh*k)))
  return res

CT=Image.open("CT-processed.jpg").convert("L")
T2=Image.open("T2.jpg").convert("L")
CTw, CTh =CT.size
T2w, T2h = T2.size
maxcnt = 0
cnt = 0
k = 1
test = []

while cnt >= maxcnt:
  if len(test) > 25:
    break
  maxcnt = cnt
  CT_p = resizing(CT, k)
  w_p, h_p = CT_p.size
  maxw = max(w_p, T2w)
  maxh = max(h_p, T2h)
  padCT = pad_img(maxw, maxh, CT_p)
  padT2 = pad_img(maxw, maxh, T2)
  mCT = np.asarray(padCT)
  mT2 = np.asarray(padT2)
  cnt = overlap(mCT, mT2, maxw, maxh)
  k += 0.01
  test.append(cnt)

print k - 0.02
print test
CTprocessed = resizing(CT, k-0.02)

fig=plt.figure()
fig.add_subplot(121)
plt.title("T2")
plt.imshow(T2,cmap=plt.get_cmap("gray"))
fig.add_subplot(122)
plt.title("CT_p")
plt.imshow(CTprocessed,cmap=plt.get_cmap("gray"))
plt.show()