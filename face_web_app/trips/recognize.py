import cv2
import time
import os
import numpy 
import csv
from PIL import Image
# Append the images with the extension .sad into image_paths
#def uselbph_formale(par):
 #   cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#    faceCascade = cv2.CascadeClassifier(cascadePath)
   # lbphrecognizer = cv2.createLBPHFaceRecognizer()
  #  pathlbph=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/lbph_for_male.xml'
 #   lbphrecognizer.load(pathlbph)
#    path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
   # predict_image_pil = Image.open(path).convert('L')
  #  predict_image = numpy.array(predict_image_pil, 'uint8')
 #   faces = faceCascade.detectMultiScale(predict_image)
#    for (x, y, w, h) in faces:
    #        pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
   #         repic=pic.resize((50,50),Image.ANTIALIAS )
  #          repic_img=numpy.array(repic, 'uint8')
 #   nbr_predicted, conf = lbphrecognizer.predict(repic_img[:])
#return nbr_predicted
def uselbph(par):
    cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    lbphrecognizer = cv2.createLBPHFaceRecognizer()
    pathlbph=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/lbph.xml'
    lbphrecognizer.load(pathlbph)
    path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
    predict_image_pil = Image.open(path).convert('L')
    predict_image = numpy.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
         if (len(predict_image[y: y + h])!=0 and len(predict_image[x: x + w])!=0):
               
            pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
            repic=pic.resize((50,50),Image.ANTIALIAS )
            repic_img=numpy.array(repic, 'uint8')
            nbr_predicted, conf = lbphrecognizer.predict(repic_img[:])
         else:
            nbr_predicted=-1
         return nbr_predicted
         
def useigen(par):
    cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    eigenrecognizer = cv2.createEigenFaceRecognizer()
    patheigen=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/eigen.xml'
    eigenrecognizer.load(patheigen)
    path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
    predict_image_pil = Image.open(path).convert('L')
    predict_image = numpy.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        if (len(predict_image[y: y + h])!=0 and len(predict_image[x: x + w])!=0): 
            pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
            repic=pic.resize((50,50),Image.ANTIALIAS )
            repic_img=numpy.array(repic, 'uint8')
            nbr_predicted, conf = eigenrecognizer.predict(repic_img[:])
        else:
            nbr_predicted=-1
        return nbr_predicted
#def useigen_formale(par):
 ##   cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
#    faceCascade = cv2.CascadeClassifier(cascadePath)
 ##   eigenrecognizer = cv2.createEigenFaceRecognizer()
 #   patheigen=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/eigen_for_male.xml'
#    eigenrecognizer.load(patheigen)
 #   path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
  #  predict_image_pil = Image.open(path).convert('L')
   # predict_image = numpy.array(predict_image_pil, 'uint8')
   # faces = faceCascade.detectMultiScale(predict_image)
   # for (x, y, w, h) in faces:
    #        pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
     #       repic=pic.resize((50,50),Image.ANTIALIAS )
      #3      repic_img=numpy.array(repic, 'uint8')
   # nbr_predicted, conf = eigenrecognizer.predict(repic_img[:])
   # return nbr_predicted
#def usefisher_forfemale(par):
  #  cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
  #  faceCascade = cv2.CascadeClassifier(cascadePath)
  #  fisherrecognizer = cv2.createFisherFaceRecognizer()
  #  pathfisher=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/fisher_for_female.xml'
  #  fisherrecognizer.load(pathfisher)
   # path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
  #  predict_image_pil = Image.open(path).convert('L')
  #  predict_image = numpy.array(predict_image_pil, 'uint8')
  #  faces = faceCascade.detectMultiScale(predict_image)
  #  for (x, y, w, h) in faces:
  #          pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
  #          repic=pic.resize((50,50),Image.ANTIALIAS )
  ##          repic_img=numpy.array(repic, 'uint8')
  #  nbr_predicted, conf = fisherrecognizer.predict(repic_img[:])
 #   return nbr_predicted
#def usefisher_formale(par):
 #   cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
  #  faceCascade = cv2.CascadeClassifier(cascadePath)
   # fisherrecognizer = cv2.createFisherFaceRecognizer()
  #  pathfisher=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/fisher_for_male.xml'
  #  fisherrecognizer.load(pathfisher)
 #   path=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
 #   predict_image_pil = Image.open(path).convert('L')
 #   predict_image = numpy.array(predict_image_pil, 'uint8')
 #   faces = faceCascade.detectMultiScale(predict_image)
  #  for (x, y, w, h) in faces:
  #          pic=Image.fromarray(predict_image[y: y + h, x: x + w],'L')
 #           repic=pic.resize((50,50),Image.ANTIALIAS )
 #           repic_img=numpy.array(repic, 'uint8')
  #  nbr_predicted, conf = fisherrecognizer.predict(repic_img[:])
   # return nbr_predicted


























   
