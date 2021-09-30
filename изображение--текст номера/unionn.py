import cv2
import numpy as np
# import imutils
import matplotlib.pyplot as plt
import math
import keras
from keras.models import load_model
import json
import os
import sys

if len(sys.argv)>1:
	img_Name = sys.argv[1]
else:
	img_Name = 'input.jpg'


model = load_model('ultimate.h5')

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

def plateDetect():
	img_org = cv2.imread(img_Name)

	size = np.shape(img_org)
	# if size[0] <= 776:
	# img_org = imutils.resize(img_org , 900)

	img_org2 = img_org.copy()

	img_bw = cv2.cvtColor(img_org , cv2.COLOR_BGR2GRAY)
	cv2.imwrite('outp/0img_bw.jpg', img_bw)

	img_bilin = cv2.bilateralFilter(img_bw, 11, 17, 17)
	cv2.imwrite('outp/1img_bilin.jpg', img_bilin)

	ret3,img_thr = cv2.threshold(img_bilin,100,255,cv2.THRESH_BINARY)
	cv2.imwrite('outp/2img_thr.jpg', img_thr)

	img_edg  = cv2.Canny(img_thr ,100,200)
	cv2.imwrite('outp/3cn_edge.jpg' , img_edg)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	cv2.imwrite('outp/4img_kernel.jpg',kernel)
	closed = cv2.morphologyEx(img_edg, cv2.MORPH_CLOSE, kernel)
	cv2.imwrite('outp/5img_closed.jpg',closed)


	# img_dil = cv2.dilate(img_edg, closed, iterations = 1)
	# cv2.imwrite('outp/6img_dil.jpg',img_dil)


	#if  you  are  using  opencv 2.X then  make  sure  to  remove  "something_else " variable  from  list  below

	# (somethig_else,contours) = cv2.findContours(img_dil.copy(), 1, 2)
	(cnts) = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]
	# cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

	screenCnt = None

	for c in cnts:
		# print(c)
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break


	mask = np.zeros(img_bw.shape, dtype=np.uint8)
	roi_corners = np.array(screenCnt ,dtype=np.int32)
	ignore_mask_color = (255,)*1
	cv2.fillPoly(mask, roi_corners , ignore_mask_color)
	cv2.drawContours(img_org, [screenCnt], -40, (100, 255, 100), 2)
	# cv2.imshow('original  image with boundry' , img_org)
	cv2.imwrite('outp/plate_detedted.jpg',img_org)


	ys =[screenCnt[0,0,1] , screenCnt[1,0,1] ,screenCnt[2,0,1] ,screenCnt[3,0,1]]
	xs =[screenCnt[0,0,0] , screenCnt[1,0,0] ,screenCnt[2,0,0] ,screenCnt[3,0,0]]

	ys_sorted_index = np.argsort(ys)
	xs_sorted_index = np.argsort(xs)

	x1 = screenCnt[xs_sorted_index[0],0,0]-2
	x2 = screenCnt[xs_sorted_index[3],0,0]+2

	y1 = screenCnt[ys_sorted_index[0],0,1]-2
	y2 = screenCnt[ys_sorted_index[3],0,1]+2


	img_plate = img_org2[y1:y2 , x1:x2]

	cv2.imwrite('outp/number_plate.jpg',img_plate)
	splitNumber()


def splitNumber():
	img = cv2.imread('outp/number_plate.jpg')

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.bilateralFilter(img, 11, 17, 17)
	ret3,img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)


	# In[3]:


	img_edg  = cv2.Canny(img ,100,200)


	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


	# In[5]:
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
	# cv2.imwrite('outp/4img_kernel.jpg',kernel)
	img_edg = cv2.morphologyEx(img_edg, cv2.MORPH_CLOSE, kernel)
	# cv2.imwrite('outp/5img_closed.jpg',closed)


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


	clearFolder()
	for cnt in cnts:
	    x,y,w,h = cv2.boundingRect(cnt)
	    cv2.rectangle(check,(x,y),(x+w,y+h),(0,255,0),1)
	    crop_img = img[y:y+h,x:x+w]
	#     print(len(crop_img),len(crop_img[0]),w,h)
	    buf = cv2.resize(crop_img , (30,50)) 
	    path = 'outp_num/'+str(coordX.index(x)).rjust(2,'0')+'_'+str(int(iterat))+'.jpg'
	    # print(path)
	    cv2.imwrite(path,buf)
	    # print(iterat)
	    iterat += 1

	cv2.imwrite('outp/check.jpg',check)
	predictPlate()


def clearFolder():
	files = os.listdir(path="outp_num")
	files.sort()
	for file in files:
		os.remove('outp_num/'+ file)



def predictPlate():
	files = os.listdir(path="outp_num")
	files.sort()
	dataset = []
	for path in files:
	    img = cv2.imread('outp_num/'+path,0)
	    dataset.append(np.array(img).ravel())

	predict = model.predict([dataset])
	letter = []
	for answer in predict:
		letter.append(np.where(answer == np.max(answer))[0][0])
	char_by_num = {
	    0 : '0',
	    1 : '1',
	    2 : '2',
	    3 : '3',
	    4 : '4',
	    5 : '5',
	    6 : '6',
	    7 : '7',
	    8 : '8',
	    9 : '9',
	    10 : 'a',
	    11 : 'b',
	    12 : 'c',
	    13 : 'e',
	    14 : 'h',
	    15 : 'k',
	    16 : 'm',
	    17 : 'o',
	    18 : 'p',
	    19 : 't',
	    20 : 'x',
	    21 : 'y'
	}
	output = ''
	for i,cha in enumerate(letter):
	    if(predict[i][cha]>0.5):
	        output += char_by_num[cha]
	file_out = open('outp.txt', 'w')
	file_out.write(output.upper())
	file_out.close()
	print(output.upper())

plateDetect()