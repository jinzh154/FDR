import os
import numpy as np
import csv
import sys
import tensorflow as tf
import numpy.linalg as la

from keras.models import Model
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils

from art.attacks import ProjectedGradientDescent
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from utilities import *


if __name__ == '__main__':
    #0. Creat output file
    csv_name = 'csv_out/results_MNIST_LeNet_4threats.csv'
    make_result_file_class(csv_name)
    #1.Load keras model
    directory = "models/mnist_lenet/"
    M_list=['LeNet-1.h5','LeNet-4.h5','LeNet-5.h5']
    Idx_list=[-1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    
    for i,file_name in enumerate(M_list):
        string = directory + file_name
        model = load_model(string)
        print('model name',file_name)
      #2.import adv
        model_name=os.path.splitext(file_name)[0]
        attack_list=['Contrast','PGD','RT','Gaussian']
        for i,att_name in enumerate(attack_list):

            if att_name=='Contrast':
                eps_range=np.linspace(0.1,0.8,20)
                result(x_train, y_train,x_test, y_test,model,model_name,Idx_list,att_name,eps_range,csv_name)
            
            elif att_name=='PGD':
                eps_range=np.linspace(0.01,0.1,20)
                result(x_train, y_train,x_test, y_test,model,model_name,Idx_list,att_name,eps_range,csv_name)

            elif att_name=='Gaussian':
                eps_range=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                result(x_train, y_train,x_test, y_test,model,model_name,Idx_list,att_name,eps_range,csv_name)

            elif att_name=='RT':
                eps_range=np.linspace(0,40,20)
                result(x_train, y_train,x_test, y_test,model,model_name,Idx_list,att_name,eps_range,csv_name)
