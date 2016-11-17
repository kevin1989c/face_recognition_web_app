import cv2
import time
import os
import numpy 
import csv
import sys
from PIL import Image
#Haar face detector
def detect(pic):
    image = numpy.array(pic.convert('L'), 'uint8')
    
    haarcascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    facedetect = cv2.CascadeClassifier(haarcascadePath)
    faces = facedetect.detectMultiScale(image)
    for (x, y, w, h) in faces:
            #print len(image[y: y + h])
           # print len(image[x: x + w])
           # print('.............')
            pic=Image.fromarray(image[y: y + h, x: x + w],'L')
            if (len(image[y: y + h])!=0 and len(image[x: x + w])!=0):
              repic=pic.resize((50,50),Image.ANTIALIAS )          
              return repic
            #if(len(image[y: y + h])<(len(image[x: x + w])/10) or len(image[x: x + w])<(len(image[y: y + h])/10)):
              #return 0
#use fisherface to recognize gender
def gender(par):
 fisherrecognizer = cv2.createFisherFaceRecognizer()
 path_g_check=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
 input_pic=Image.open(path_g_check)
 detected_face=detect(input_pic)
 if (detected_face!=None) and detected_face != 0: 
     repic_img=numpy.array(detected_face, 'uint8')
     xmlpath=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/genderclass.xml'
     fisherrecognizer.load(xmlpath)
     predicted_fisher, conf2_fisher = fisherrecognizer.predict(repic_img[:])
     return predicted_fisher
  
  













