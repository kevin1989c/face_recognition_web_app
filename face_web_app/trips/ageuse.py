import os
import sys
import cPickle
import csv
import numpy
from PIL import Image
import cv2
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv


def detect(pic):
    image = numpy.array(pic.convert('L'), 'uint8')
    cascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    faces = faceCascade.detectMultiScale(image)
    for (x, y, w, h) in faces:
          if (len(image[y: y + h])!=0 and len(image[x: x + w])!=0): 
            pic=Image.fromarray(image[y: y + h, x: x + w],'L')
            repic=pic.resize((50,50),Image.ANTIALIAS )
          else:
            repic=None
          return repic

def load_params():
    path_age=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+'/age_cnn.pkl'
    fr=open(path_age,'rb')

    layer0_params=cPickle.load(fr)
    layer1_params=cPickle.load(fr)
    layer2_params=cPickle.load(fr)
    layer3_params=cPickle.load(fr)
    fr.close()
    return layer0_params,layer1_params,layer2_params,layer3_params





def load_data(file_path, img_h, img_w):
     faces=numpy.zeros((1,img_h*img_w))   
     i=0
     img1=Image.open(file_path)
     img2=img1.convert('L')
     img=detect(img2)
     if (img!=None):
         img_ndarray=numpy.asarray(img,dtype='float64')/256
         faces[i]=numpy.ndarray.flatten(img_ndarray[:])
     else:
         faces=None
  
     return faces

#same structure as the training code
class LogisticRegression(object):
    def __init__(self, input, params_W, params_b, n_in, n_out):
        self.W = params_W
        self.b = params_b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, input, params_W,params_b, n_in, n_out,
                 activation=T.tanh):
        self.input = input
        self.W = params_W
        self.b = params_b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

	
class LeNetConvPoolLayer(object):
    def __init__ (self, input, filter_w, filter_h, filter_num, img_w, img_h, input_feature,batch_size, poolsize,params_W,params_b):

        self.input = input
        self.W = params_W
        self.b = params_b
        
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=(filter_num, input_feature, filter_h, filter_w),
            image_shape=(batch_size,input_feature,img_h,img_w)
        )
        
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]


#combine all the layers
def use_CNN(file_path,filter_num1, filter_num2,filter_h,filter_w,img_h,img_w,poolsize,input_feature,total_image_num,class_num, count_incorrect=0.0): 
    
   
    faces=load_data(file_path, img_h, img_w)


  

    layer0_params,layer1_params,layer2_params,layer3_params=load_params()
    
    x = T.matrix('x')  


    layer0_input = x.reshape((total_image_num, input_feature, img_h, img_w))
    layer0 = LeNetConvPoolLayer(
        input=layer0_input,
        params_W=layer0_params[0],
        params_b=layer0_params[1],
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num1, 
        img_h=img_h,         
        img_w=img_w, 
        input_feature=input_feature,
        batch_size=total_image_num, 
        poolsize=poolsize
    )
    layer1_input=layer0.output
    layer1_image_h=(img_h-filter_h+1)/2
    layer1_image_w=(img_w-filter_w+1)/2
    layer1 = LeNetConvPoolLayer(
        input=layer1_input,
        params_W=layer1_params[0],
        params_b=layer1_params[1],
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num2,  
        img_h=layer1_image_h, 
        img_w=layer1_image_w, 
        input_feature=filter_num1,
        batch_size=total_image_num,  
        poolsize=poolsize
    )

    
    layer2_input = layer1.output.flatten(2)
    layer2_image_h=(layer1_image_h-filter_h+1)/2
    layer2_image_w=(layer1_image_w-filter_w+1)/2

    layer2 = HiddenLayer(
        input=layer2_input,
        params_W=layer2_params[0],
        params_b=layer2_params[1],
        n_in=filter_num2 *  layer2_image_h * layer2_image_w,
        n_out=500,      
        activation=T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, params_W=layer3_params[0],params_b=layer3_params[1],n_in=500, n_out=class_num)  

 
     

    f = theano.function(
        [x],    
        layer3.y_pred
    )
    
    if (faces!=None):
      pred = f(faces)
    else:
      pred=[-1]
    #predict result
    return pred[0]
    

def face_age(par):
        file_path_t=os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir))+"%s"%(str(par))
	pred=use_CNN(
        file_path=file_path_t,
        filter_num1=10,
        filter_num2=20,
        filter_h=7,
        filter_w=7,
        img_h=50,
        img_w=50,
        poolsize=(2,2),
        total_image_num=1,
        class_num=6,
        input_feature=1)

        return pred
