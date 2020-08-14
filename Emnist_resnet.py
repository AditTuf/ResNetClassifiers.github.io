# save the final model to file
#from tensorflow.keras.datasets.mnist import load_data
import emnist as em
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense ,Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.constraints import Constraint ,UnitNorm
from tensorflow.keras.preprocessing.image import array_to_img
import tensorflow as tf
from sklearn.utils import shuffle

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

 
  
# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY) = em.extract_training_samples('letters')
    trainX, trainY = shuffle(trainX, trainY)
    (testX, testY) = em.extract_test_samples('letters')
    # reshape dataset to have a single channel
    #(trainX,trainY),(testX,testY)=load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    #trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    tX=[]
    tY=[]
    # print("tX.shape",tX.shape)
    # print("tY.shape",tY.shape)
    # #print(trainy[0])
    shot = 300
    ctr = [shot]*27
    for i in range(len(trainY)):
      label=trainY[i]
      ctr[label]=ctr[label]-1
      if(ctr[label]>0):
        tX.append(trainX[i])
        tY.append(trainY[i])
    print("tX.shape",len(tX))
    tY = to_categorical(tY)
    
    # print("tY.shape",tY.shape)

    return tX, tY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train=np.array(train)
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    #train_norm = train_norm / 255.0
    #test_norm = test_norm / 255.0
    test_norm = (test_norm - 127.5) / 127.5
    train_norm = (train_norm -127.5) /127.5
    # return normalized images
    return train_norm, test_norm

 
def resnet_block(n_filters, input_layer):
  g = BatchNormalization(axis=-1)(input_layer)
  g = LeakyReLU(alpha=0.2)(g)
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer='Orthogonal')(g)
  # second convolutional layer
  g = BatchNormalization(axis=-1)(g)
  g = LeakyReLU(alpha=0.2)(g)
  g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer='Orthogonal')(g)
  # second convolutional layer
  # concatenate merge channel-wise with input layer
  g = Concatenate()([g, input_layer])
  return g
 
# define cnn model
def define_model(image_shape=(28,28,1),n_resnet=1):
  in_src_image = Input(shape=image_shape)
  
  gen =resnet_block(16,in_src_image)
  gen =resnet_block(32,gen)
  
  c1 = Conv2D(32, (3, 3), strides=(2,2), padding='same', kernel_initializer='Orthogonal',kernel_constraint=UnitNorm())(gen)
  c1b = BatchNormalization()(c1, training=True)
  gen=LeakyReLU(alpha=0.2)(c1b) # 14x14
  
  gen =resnet_block(32,gen)

  c1 = Conv2D(64, (3, 3), strides=(2,2), padding='same', kernel_initializer='Orthogonal',kernel_constraint=UnitNorm())(gen)
  c1b = BatchNormalization()(c1, training=True)
  gen=LeakyReLU(alpha=0.2)(c1b) #7x7

  gen =resnet_block(64,gen)
  #gen =resnet_block(128,gen)

  fe = Flatten()(gen)
  out1 = Dense(50)(fe)
  out1=LeakyReLU(alpha=0.2)(out1)
  model_emb=Model(in_src_image,out1)
    # compile model
  return model_emb
 
def define_classifier(d_model,image_shape=(28,28,1)):
  in_src_image=Input(shape=image_shape)
  ip =d_model(in_src_image)
  #ip =tf.keras.backend.l2_normalize(ip, axis=0)

  ip = Dropout(0.4)(ip)
  out1=Dense(27, activation='softmax')(ip) #..................Changed number of classes here for Mnist
  opt = Adam(lr=0.005, beta_1=0.9)
  model_class=Model(in_src_image,out1)
  model_class.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model_class
 
def train(d_model,c_model):
    model_full=c_model
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print("Print training on ",trainX.shape[0])
    # trainX = np.concatenate([trainX,testX], axis=0)
    # trainY = np.concatenate([trainY,testY], axis=0)

    model_full.fit(trainX, trainY, epochs=8, batch_size=64)
    return 
  
# entry point, run the test harness
image_shape=(28,28,1)
d_model=define_model(image_shape)
c_model=define_classifier(d_model,image_shape)
train(d_model,c_model)


d_model.save('Emnist.h5')
c_model.save('Classifier.h5')
plot_model(c_model, to_file='gan_model_plot.png', show_shapes=True, show_layer_names=True)
#c_model.save('final_model_14.h5')
from tensorflow.keras.models import load_model
def run_test_harness():
    # load dataset
    # load model
  print("Running test harness")
  trainX, trainY, testX, testY = load_dataset()
  # prepare pixel data
  trainX, testX = prep_pixels(trainX, testX)
  print("Examples in test set",testX.shape[0])
  model = c_model
  # evaluate model on test dataset
  _, acc = model.evaluate(testX, testY, verbose=0)
  print('> %.3f' % (acc * 100.0))
 
run_test_harness()

# Dropout 0.9
