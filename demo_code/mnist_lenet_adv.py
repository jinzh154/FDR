import os
import numpy as np


from keras.models import load_model

from keras.datasets import mnist
from keras.utils import np_utils

from art.attacks import FastGradientMethod,CarliniLInfMethod,CarliniL2Method,ProjectedGradientDescent
from art.attacks import SpatialTransformation
from art.classifiers import KerasClassifier
from utilities import *



def gen_adv(file_name):
    model_base='MNIST_lenet'
    model_name=os.path.splitext(file_name)[0]

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    x=x_test
    label=y_test
    directory = "models/mnist_lenet/"
    string = directory + file_name
    model = load_model(string)
    print('model name',string)
    classifier = KerasClassifier(model=model, clip_values=(0, 255))
    attack_list=['Contrast','PGD','RT','Gaussian']
    for i,att_name in enumerate(attack_list):
        if att_name=='Contrast':
            eps_range=np.linspace(0.1,0.8,20)
            for eps in eps_range:
                x_adv=contrast(x, eps=eps) 
                #save adversarial inputs to .npy
                save_path = 'adv/'
                adv_name=model_name+'_'+att_name+'_'+str(eps)
                completeName = os.path.join(save_path, adv_name+'.npy')
                np.save( completeName, x_adv)
                eval_acc(x,x_adv,y_test,model)
            print('save adv for {} completed!'.format(attack_list[i]))

        elif att_name=='PGD':
            eps_range=np.linspace(0.01,0.1,20)
            for eps in eps_range:
                attack = ProjectedGradientDescent(classifier, eps=eps, eps_step=eps/5, max_iter=10, targeted=False,
                                  num_random_init=True)
                x_adv=save_adv_eps(x, att_name, attack, eps, model_name)
                eval_acc(x,x_adv,y_test,classifier)
            print('save adv for {} completed!'.format(attack_list[i]))

        elif att_name=='RT':
            rotation_range=np.linspace(0,40,20)
            for max_rotation in rotation_range:
                attack = SpatialTransformation(classifier,max_translation=5,num_translations=5,max_rotation=int(max_rotation),num_rotations=int(max_rotation)+1)      
                x_adv=save_adv_eps(x, att_name, attack, max_rotation, model_name)
                eval_acc(x,x_adv,y_test,classifier)
            print('save adv for {} completed!'.format(attack_list[i]))

        elif att_name=='Gaussian':
            eps_range=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            for eps in eps_range:
                x_adv=gaussian_noise(x,severity=eps) 
                #save adversarial inputs to .npy
                save_path = 'adv/'
                adv_name=model_name+'_'+att_name+'_'+str(eps)
                completeName = os.path.join(save_path, adv_name+'.npy')
                np.save( completeName, x_adv)
                eval_acc(x,x_adv,y_test,model)
            print('save adv for {} completed!'.format(attack_list[i]))
                    
if __name__ == '__main__':
    gen_adv(file_name="LeNet-5.h5")

