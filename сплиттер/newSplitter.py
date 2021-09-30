#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imutils
import math
# import shutil

# shutil.rmtree('outp_num')


# In[2]:
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

img = cv2.imread('outp/number_plate.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('outp/split_1_gray.jpg',img)
img = cv2.bilateralFilter(img, 11, 17, 17)
cv2.imwrite('outp/split_2_bilater.jpg',img)
ret3,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
cv2.imwrite('outp/split_3_thresh.jpg',img)



# In[3]:


img_edg  = cv2.Canny(img ,100,200)


# In[4]:


img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# In[5]:
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# cv2.imwrite('outp/4img_kernel.jpg',kernel)
img_edg = cv2.morphologyEx(img_edg, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite('outp/5img_closed.jpg',closed)
cv2.imwrite('outp/split_5_edged.jpg',img_edg)


(cnts) = cv2.findContours(img_edg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:20]


# In[7]:

check = img.copy()
iterat = 0
coordX = []


for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    coordX.append(x)
coordX.sort()

for cnt in cnts:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(check,(x,y),(x+w,y+h),(0,255,0),1)
    crop_img = img[y:y+h,x:x+w]
#     print(len(crop_img),len(crop_img[0]),w,h)
    buf = cv2.resize(crop_img , (30,50)) 
    path = 'outp_num/'+str(coordX.index(x)).rjust(2,'0')+'_'+str(int(iterat))+'.jpg'
    print(path)
    cv2.imwrite(path,buf)
    print(iterat)
    iterat += 1

cv2.imwrite('outp_num/check.jpg',check)
# In[ ]:




