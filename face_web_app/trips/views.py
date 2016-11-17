from __future__ import unicode_literals
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from datetime import datetime
from .models import IMG
from .models import Celebr
from .models import Celebr_new
import recognize as rc
from django.shortcuts import render_to_response
import os
import OPENCVTRAIN as tr
from django.contrib import auth 
import pdb
from django import forms
import csv
import re
import codecs
import genderuse
import ageuse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def index(request):
 
    return render_to_response('index.html',locals())

def home(request):
   post_list = Celebr.objects.all()
  
   return render(request,'home.html',{'post_list':post_list,'src':src,})


def hello_world(request):
    return render(request, 'hello_world.html',{'current_time':str(datetime.now()),})


def search_form(request):
	return render_to_response('search_form.html')

def adddata(request):
        csvpath=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/index.csv'
        csvfile=codecs.open(csvpath,'r','utf-8')
        csvdata=csv.reader(csvfile)
        
        csvpath_new=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/newadd.csv'
        csvfile_new=open(csvpath_new)
        csvdata_new=csv.reader(csvfile_new)
        i=1
        for n,g,c in csvdata:
             Celebr.objects.get_or_create(name=n, gender=g, content=c, mem_id=i, img="upload/"+str(n)+".jpg")
             i=i+1
        for no,new in csvdata_new:
             Celebr_new.objects.get_or_create(name=str(new),img="new/"+str(new)+".jpg")
        sti='done'
        return render(request, 'uploading.html', {'sti':sti,})
        

def train(request):
        celelist=Celebr.objects.all()
        newlist=Celebr_new.objects.all()
        stitu=tr.trainpic(celelist,newlist)
        return render(request, 'uploading.html', {'stitu':stitu,})


def search(request):  

        post_list = Celebr.objects.all()
        imgs = IMG.objects.all()
        imglast=imgs[len(imgs)-1]
  
	
        predi_lbph=rc.uselbph(imglast.img.url)
        predi_eigen=rc.useigen(imglast.img.url)
        predi_gender=genderuse.gender(imglast.img.url)
        predi_age=ageuse.face_age(imglast.img.url)
        if (predi_gender==0):
           predi_gender="Predicted_Gender: Female"
        if (predi_gender==1):
           predi_gender="Predicted_Gender: Male" 
        if (predi_age==-1):
           predi_age="None"
        if (predi_age==0):
           predi_age="Pridicted_age: from 0 to 7 "
        if (predi_age==1):
           predi_age="Pridicted_age: from 8 to 14" 
        if (predi_age==2):
           predi_age="Pridicted_age: from 15 to 24"
        if (predi_age==3):
           predi_age="Pridicted_age: from 25 to 47" 
        if (predi_age==4):
           predi_age="Pridicted_age: from 48 to 59"
        if (predi_age==5):
           predi_age="Pridicted_age: from 60 to ..."
            
        #else:
        #  predi_fisher=rc.usefisher_forfemale(imglast.img.url)
          #predi_lbph=rc.uselbph_forfemale(imglast.img.url)
          #predi_eigen=rc.useigen_forfemale(imglast.img.url)
        predi_1=-2
        predi_2=-2
        if (predi_lbph<0 and predi_eigen<0):
          predi_1=-1
          predi_2=-1

        if (predi_lbph!=predi_eigen):
          # predi_1=predi_fisher
           predi_1=predi_eigen
           predi_2=predi_lbph
        if (predi_eigen==predi_lbph):
           predi_1=predi_eigen 
       
        predi_img1=None
        predi_title1=None
        predi_content1=None
        predi_img2=None
        predi_title2=None
        predi_content2=None
       # predi_img3=None
      # predi_title3=None
       # predi_content3=None
      
        for i in post_list:
           if (i.mem_id==predi_1):
              predi_img1=i.img
              predi_title1=i.name
              predi_content1=i.content
           if (i.mem_id==predi_2):
              predi_img2=i.img
              predi_title2=i.name
              predi_content2=i.content
           #if (i.mem_id==predi_3):
             # predi_img3=i.img
             # predi_title3=i.title
             # predi_content3=i.content
        
           #fs.usefisher(request.GET['q'])

	return render(request,'home.html',{'predi_img1':predi_img1,'predi_content1':predi_content1,'predi_title1':predi_title1,'predi_img2':predi_img2,'predi_content2':predi_content2,'predi_title2':predi_title2,'imglast':imglast,'predi_age':predi_age,'predi_gender':predi_gender,})



def uploadImg(request):
    if request.method == 'POST':
        new_img = IMG(img=request.FILES.get('img'))
        new_img.save()
    return render(request, 'uploading.html')

def showImg(request):
    imgs = Celebr.objects.all()
    imgs_new=Celebr_new.objects.all()
    return render(request,'showing.html',{'imgs':imgs,'imgs_new':imgs_new,})
def showNewImg(request):
    imgs_new=Celebr_new.objects.all()
    return render(request,'showingnew.html',{'imgs_new':imgs_new,})
# Create your views here.