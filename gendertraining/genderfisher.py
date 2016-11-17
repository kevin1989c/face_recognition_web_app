import os
import sys
import time
import csv
import numpy
from PIL import Image
import cv2
import re

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
            if(len(image[y: y + h])<(len(image[x: x + w])/5) or len(image[x: x + w])<(len(image[y: y + h])/5)):
              return 0

    

images = []
labels = []  
csv_file=open('gendertrain.csv')
data=csv.reader(csv_file)
print ("adding data...")
for c1,c2,c3,c4 in data:

     if (str(c4)=='m' or str(c4)=='f'):
     
        
       if(str(c4)=='m'):
        l=1
       if(str(c4)=='f'):
        l=0

       rootDir="/home/kevin/Downloads/adienceface/faces/"+str(c1)+"/"
       c=c3+'.'
       name_pattern=re.compile(".*"+c+c2+"$")
       for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if name_pattern.match(path):
           img1=Image.open(path)
           img2=img1.convert('L')
           img=detect(img2)
           if (detect(img2)!=None) and img !=0: 
             repic_img=numpy.array(img, 'uint8')   
             images.append(repic_img[:])
             labels.append(l)
print("in training...")
fisherrecognizer = cv2.createFisherFaceRecognizer()
fisherrecognizer.train(images, numpy.array(labels))
fisherrecognizer.save("./genderclass.xml")




