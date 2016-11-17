import os
import sys
import time
import csv
import numpy
from PIL import Image
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import cPickle  
import cv2
import re


def detect(pic):
    image = numpy.array(pic.convert('L'), 'uint8')
    
    haarcascadePath = "/usr/local/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    facedetect = cv2.CascadeClassifier(haarcascadePath)
    faces = facedetect.detectMultiScale(image)
    for (x, y, w, h) in faces:
            print len(image[y: y + h])
            print len(image[x: x + w])
            print('.............')
            pic=Image.fromarray(image[y: y + h, x: x + w],'L')
            if (len(image[y: y + h])!=0 and len(image[x: x + w])!=0):
              repic=pic.resize((50,50),Image.ANTIALIAS )          
              return repic
            if(len(image[y: y + h])<(len(image[x: x + w])/5) or len(image[x: x + w])<(len(image[y: y + h])/5)):
              return 0

def count(traincsv,validcsv,databaserootfile='/home/kevin/Downloads/adienceface/'):
  total_t=0
  total_v=0
  csv_file=open(traincsv)
  data=csv.reader(csv_file)

  csv_file_d=open(validcsv)
  data_d=csv.reader(csv_file_d)
  for dq,dw,de,dr in data_d:
    if (dr.isdigit()):
       rootDir=str(databaserootfile)+"faces/"+str(dq)+"/"
       d=de+'.'
       name_pattern=re.compile(".*"+d+dw+"$")
       for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if name_pattern.match(path):
           img1=Image.open(path)      
           img=detect(img1)
           if (detect(img1)!=None) and img !=0:
             total_v=total_v+1

  for cq,cw,ce,cr in data:
      if (cr.isdigit()):

       rootDir="/home/kevin/Downloads/adienceface/faces/"+str(cq)+"/"
       c=ce+'.'
       name_pattern=re.compile(".*"+c+cw+"$")
       for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if name_pattern.match(path):
           img1=Image.open(path)      
           img=detect(img1)
           if (detect(img1)!=None) and img !=0:
            total_t=total_t+1

  return total_t,total_v



def load_data(traincsv,validcsv, img_h, img_w):
  total_t,total_v=count(traincsv,validcsv)
  i=0
  j=0
  csv_file=open(traincsv)
  data=csv.reader(csv_file)
  train_data=numpy.zeros((total_t,img_h*img_w))   
  train_label=numpy.zeros(total_t)




  for c1,c2,c3,c4 in data:

   if (c4.isdigit()):
       if (int(c4)<8):    
         l=0
       if (int(c4)<15 and int(c4))>=8:
         l=1
       if (int(c4)<25 and int(c4))>=15:
         l=2    
       if (int(c4)<48 and int(c4))>=25:
         l=3
       if (int(c4)<60 and int(c4))>=48:
         l=4
       if (int(c4))>=60:
         l=5            
    

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
           
             img_ndarray=numpy.asarray(img,dtype='float64')/256
             train_data[i]=numpy.ndarray.flatten(img_ndarray[:])
             train_label[i]=l
             i=i+1
           






  csv_file_d=open(validcsv)
  data_d=csv.reader(csv_file_d)
  valid_data=numpy.zeros((total_v,img_h*img_w))   
  valid_label=numpy.zeros(total_v)


  for d1,d2,d3,d4 in data_d:
      

      if (d4.isdigit()):
       if (int(d4)<8):    
         ld=0
       if (int(d4)<15 and int(d4))>=8:
         ld=1
       if (int(d4)<25 and int(d4))>=15:
         ld=2    
       if (int(d4)<48 and int(d4))>=25:
         ld=3
       if (int(d4)<60 and int(d4))>=48:
         ld=4
       if (int(d4))>=15:
         ld=5       
       rootDir_d="/home/kevin/Downloads/adienceface/faces/"+str(d1)+"/"
       d=d3+'.'
       name_pattern_d=re.compile(".*"+d+d2+"$")
       for lists_d in os.listdir(rootDir_d): 
        path_d = os.path.join(rootDir_d, lists_d) 
        if name_pattern_d.match(path_d):

           img1_d=Image.open(path_d)
           img2_d=img1_d.convert('L')
           img_d=detect(img2_d)
           if (detect(img2_d)!=None) and img_d !=0:
           
             img_ndarray_d=numpy.asarray(img_d,dtype='float64')/256
             valid_data[j]=numpy.ndarray.flatten(img_ndarray_d[:])
             valid_label[j]=ld
             j=j+1
           
          


 

 

  def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')



  train_set_x, train_set_y = shared_dataset(train_data,train_label)
  valid_set_x, valid_set_y = shared_dataset(valid_data,valid_label)
  rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
  return rval




class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
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
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    def __init__ (self, rng, input, filter_w, filter_h, filter_num, img_w, img_h, input_feature,batch_size, poolsize):
        
        self.input = input

        

        fan_in = filter_w*filter_h*input_feature
        fan_out = (filter_num*filter_h*filter_w) /numpy.prod(poolsize)

        # initialize weights with random weights
        W_shape = (filter_num, input_feature, filter_w, filter_h)
        W_bound = numpy.sqrt(1. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=W_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_num,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

       
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=(filter_num, input_feature, filter_h, filter_w),
            image_shape=(batch_size,input_feature,img_h,img_w)
        )

       
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]



def save_params(param1,param2,param3,param4):  
        write_file = open('age_cnn.pkl', 'wb')   
        cPickle.dump(param1, write_file, -1)
        cPickle.dump(param2, write_file, -1)
        cPickle.dump(param3, write_file, -1)
        cPickle.dump(param4, write_file, -1)
        write_file.close()  




def train_cnn(traincsv,validcsv,learning_rate, n_epochs,batch_size,filter_num1, filter_num2,filter_h,filter_w,class_num,img_h,img_w,poolsize,input_feature):   

    rng = numpy.random.RandomState(23455)
    
    datasets = load_data(traincsv,validcsv,img_h,img_w)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

   
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size

   
    index = T.lscalar()
    x = T.matrix('x')  
    y = T.ivector('y')



   
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 57 * 47)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (57, 47) is the size of  images.
    layer0_input = x.reshape((batch_size, input_feature, img_h, img_w))


    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num1, 
        img_h=img_h,         
        img_w=img_w, 
        input_feature=input_feature,
        batch_size=batch_size, 
        poolsize=poolsize
    )
    layer1_input=layer0.output
    layer1_image_h=(img_h-filter_h+1)/2
    layer1_image_w=(img_w-filter_w+1)/2
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer1_input,
        filter_h=filter_h,        
        filter_w=filter_w, 
        filter_num=filter_num2,  
        img_h=layer1_image_h, 
        img_w=layer1_image_w, 
        input_feature=filter_num1,
        batch_size=batch_size,  
        poolsize=poolsize
    )
    
    layer2_input = layer1.output.flatten(2)
    layer2_image_h=(layer1_image_h-filter_h+1)/2
    layer2_image_w=(layer1_image_w-filter_w+1)/2

    
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=filter_num2 *  layer2_image_h * layer2_image_w,
        n_out=500,      
        activation=T.tanh
    )

   
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=class_num)  


  
    cost = layer3.negative_log_likelihood(y)
    

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

   
    params = layer3.params + layer2.params + layer1.params + layer0.params
    
    grads = T.grad(cost, params)
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]
   
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    print '... training'
   
    patience = 100000
    patience_increase = 2  
    improvement_threshold = 0.99  
    validation_frequency = min(n_train_batches, patience / 2) 


    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
		    save_params(layer0.params,layer1.params,layer2.params,layer3.params)

                   
                    
            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i' %
          (best_validation_loss * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))




if __name__ == '__main__':



	train_cnn(traincsv='age_train.csv',validcsv='agecsv4.csv',
                  learning_rate=0.007,
                  n_epochs=5000,
                  batch_size=100,
                  filter_num1=10,
                  filter_num2=20,
                  filter_h=7,
                  filter_w=7,
                  class_num=6,
                  img_h=50,
                  img_w=50,
                  poolsize=(2,2),
                  input_feature=1)

















