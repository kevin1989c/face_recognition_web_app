import cv2
import time
import os
import numpy 
import csv
from PIL import Image
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#use utf-8 because the some of the words in csv is not in ASCII


def trainpic(cele_list,newpic_list):
    cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
   
    images = []
    labels = []
   # for csvlabel in mem_id:
       #labels.append(l)
    for i in cele_list:
      if (i.mem_id>0):
        datanew=i.name.replace('%20',' ')
        f=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"/media/upload/%s.jpg"%(str(datanew))
        #gender=uc.face_dtc(f)
        image_pil = Image.open(f).convert('L')
        image = numpy.array(image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(image)
  

 #detect face and store into arrays
        for (x, y, w, h) in faces:
          #print len(image[y: y + h])
          #print len(image[x: x + w])
          print('adding celebr_index into arrays: processing..........................')
          if (len(image[y: y + h])!=0) and (len(image[x: x + w])!=0):
            #if ((len(image[y: y + h]) > len(image[x: x + w]) and (0.5*len(image[y: y + h])<en(image[x: x + w])) or ((len(image[x: x + w]) > len(image[y: y + h]) and (0.5*len(image[x: x + w])<len(image[y: y + h])):
               pic=Image.fromarray(image[y: y + h, x: x + w],'L')
               repic=pic.resize((50,50),Image.ANTIALIAS )
               repic_img=numpy.array(repic, 'uint8')
           # print ("gender=%i"%gender)
           # if (gender==0):
            #0 is male, 1 is felmale
               images.append(repic_img[:])
               labels.append(i.mem_id)
            #else:
               #images_female.append(repic_img[:])
              # labels_female.append(labels[i])
     
    #new image, compare name with Celebr and get labels          
    for j in newpic_list:
        for i in cele_list: 
            pattern=re.compile("^.*"+i.name+".*$")  
            if pattern.match(j.name):
               newlabel=i.mem_id
   
    #detect face and store into arrays
        namemore=j.name.replace('%20',' ')
        newpath="/home/kevin/Django/mysitecopycopy/media/new/%s.jpg"%(str(namemore))
        image_new = Image.open(newpath).convert('L')
        image_array = numpy.array(image_new, 'uint8')
        faces_new = faceCascade.detectMultiScale(image_array)
        for (x, y, w, h) in faces_new:
          #print len(image_array[y: y + h])
          #print len(image_array[x: x + w])
          print('adding images_new into arrays: processing.............')
          if (len(image_array[y: y + h])!=0) and (len(image_array[x: x + w])!=0):
               picnew=Image.fromarray(image_array[y: y + h, x: x + w],'L')
               repicnew=picnew.resize((50,50),Image.ANTIALIAS )
               repic_imgnew=numpy.array(repicnew, 'uint8')

               images.append(repic_imgnew[:])
               labels.append(newlabel)



   #fisherrecognizer_forfemale =  cv2.createFisherFaceRecognizer()
   # fisherrecognizer =  cv2.createFisherFaceRecognizer()
    lbphrecognizer=cv2.createLBPHFaceRecognizer()
    eigenrecognizer=cv2.createEigenFaceRecognizer()

    
    #if (len(images_male)!=0):
       #fisherrecognizer_forfemale.train(images_male, numpy.array(labels_male))
       #fisherrecognizer_forfemale.save("./fisher_for_female.xml")
    lbphrecognizer.train(images, numpy.array(labels))
    lbphrecognizer.save("./lbph.xml")
    eigenrecognizer.train(images, numpy.array(labels))
    eigenrecognizer.save("./eigen.xml")
    #if (len(images_female)!=0):
    #fisherrecognizer.train(images, numpy.array(labels))
    #fisherrecognizer.save("./fisher.xml")
     #  lbphrecognizer_formale.train(images_female, numpy.array(labels_female))
     #  lbphrecognizer_formale.save("./lbph_for_male.xml")
     #  eigenrecognizer_formale.train(images_female, numpy.array(labels_female))
     #  eigenrecognizer_formale.save("./eigen_for_male.xml")


    return("done")






