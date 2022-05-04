import numpy as np
import csv
import os
import math
    

import numpy.linalg as la
import pandas as pd

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.models import Model
from scipy.ndimage import rotate, shift


def result(x_train, y_train,x_test, y_test,model,model_name,Idx_list,att_name,eps_range,csv_name):
    load_path = 'adv'
    for j,eps in enumerate(eps_range):
        base_name=model_name+'_'+att_name
        adv_name=base_name+'_'+str(eps)
        completeName = os.path.join(load_path, adv_name+'.npy')   
        x_adv=np.load(completeName)
        #3.calculate accuracy
        print('eps=',eps)
        acc_r=eval_acc(x_test,x_adv,y_test,model)

        #4.calculate FD
        for ii,idx in enumerate (Idx_list):
            dense_model = find_target_layer(model,idx)
            x1=x_train
            x2=x_adv
            x3=x_test
            #absolute FRD
            fd_a=calculate_fd(dense_model,x1, x2)
            #base line FRD
            fd_b=calculate_fd(dense_model,x1, x3)
            #relative FRD
            fd_r=fd_b/fd_a
            #5. write results into a csv file
            write_to_file(model_name,acc_r,fd_a,fd_r,att_name,eps,idx,csv_name)

def make_result_file(file_name="results.csv"):
    if not os.path.exists(file_name):
        print("made a new results file")
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Model","Acc_r","FRD_a","FRD_r","Attack","Strength","Layer"])

def write_to_file(model,acc,fd_a,fd_r,attack,eps,layer, csv_file):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [model,acc,fd_a,fd_r,attack,eps,layer])

        
# calculate frechet distance
def calculate_fd(model, images1, images2):
  act1 = model.predict(images1)
  act2 = model.predict(images2)
# calculate mean and covariance statistics
  mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
  mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
# calculate sum squared difference between means
  ssdiff = np.sum((mu1 - mu2)**2.0)
# calculate sqrt of product between cov
  covmean = sqrtm(sigma1.dot(sigma2))
# check and correct imaginary numbers from sqrt
  if iscomplexobj(covmean):
    covmean = covmean.real
# calculate score
  T=trace(sigma1 + sigma2 - 2.0 * covmean)
  fd = ssdiff + T
  return fd

def contrast(x,eps):
  min_,max_=(0,1)
  target=(max_+min_)/2
  x_pert=x*(1-eps)+eps*target
  
  return x_pert

def bright(x,br):
  min_,max_=(0,1)
  x_pert=x+br
  x_pert=np.clip(x_pert,min_,max_)
  return x_pert

def rotation_translation(x,severity=1):
    x=np.array(x)/255
    trans=2
    rot=np.linspace(1,30,20)[severity - 1]
    x_adv = shift(x, [0, trans, trans, 0])
    x_adv = rotate(x_adv, angle=rot, axes=(1, 2), reshape=False)
    return np.clip(x_adv,0,1)*255

def gaussian_noise(x, severity=1):
    #for lenet5-MNIST
    c=np.linspace(0.001,0.005,20)[severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def save_adv_eps(x_normal,att_name,attack,eps,model_name):
    x_adv=attack.generate(x_normal)
    save_path = 'adv/'
    adv_name=model_name+'_'+att_name+'_'+str(eps)
    completeName = os.path.join(save_path, adv_name+'.npy')  
    np.save( completeName, x_adv)
    return x_adv

def eval_acc(x_normal,x_adv,y_test,classifier):
    y = classifier.predict(x_normal)
    y_pred = classifier.predict(x_adv)
    acc_orig=np.mean(np.equal(np.argmax(y, 1), np.argmax(y_test, 1)))
    acc=np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(y_test, 1)))
    print('acc_orig',acc_orig)
    print('acc_adv',acc)
    return acc/acc_orig

def find_target_layer(model,idx):
    outputs = model.layers[idx].output
    dense_model = Model(inputs=model.input, outputs=outputs)
    dense_model.compile(optimizer='adam', loss='mse')
    dense_model.set_weights(dense_model.get_weights())
    return dense_model
##################################previous code###################################################

