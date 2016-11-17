import cv2
import time
import os
import numpy 
import csv
from PIL import Image
# For face detection we will use the Haar Cascade provided by OpenCV.
#cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)



def trainpic(data,mem_id):
    images = []  
    labels = []
    for csvdata in data:
        csvdata_="/home/kevin/Django/mysitecopy%s"%(str(csvdata))
        image_pil = Image.open(csvdata_).convert('L')
        image = numpy.array(image_pil, 'uint8') 
        images.append(image[:])

    for csvlabel in mem_id:
    
        l=int(csvlabel)  
        labels.append(l)
        
    
    lbphrecognizer = cv2.createLBPHFaceRecognizer()
    eigenrecognizer =  cv2.createEigenFaceRecognizer()
    fisherrecognizer =  cv2.createFisherFaceRecognizer()
    lbphrecognizer.train(images, numpy.array(labels))
    lbphrecognizer.save("./LBPH.xml")
    fisherrecognizer.train(images, numpy.array(labels))
    fisherrecognizer.save("./fisher.xml")
    eigenrecognizer.train(images,numpy.array(labels))
    eigenrecognizer.save("./eigen.xml")
    return("done")

