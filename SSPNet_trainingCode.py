# -*- coding: utf-8 -*-
# Submission for Medical Image Analysis 
# "SSPNet: An interpretable 3D-CNN 
# for classification of schizophrenia 
# using phase maps 
# of resting-state complex-valued fMRI data"

import sklearn
from keras.utils import to_categorical
import pickle
from keras.models import Sequential
from keras.layers import Conv3D, Activation, MaxPooling3D, Flatten, Dense, BatchNormalization
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint

import h5py
import numpy as np 
import os
os.chdir('E:/code&data/')   # set path

############################################################################
##-----------------------------load data----------------------------------##
############################################################################
data = h5py.File('./SSPsamples.mat')
x = data['x'][:]  # samples
x=np.transpose(x)
x=np.swapaxes(x,3,4);x=np.swapaxes(x,2,3);x=np.swapaxes(x,1,2)

subject_num = x.shape[0]
order_num = x.shape[1]
img_l = x.shape[2]       
img_h = x.shape[3]
img_w = x.shape[4]

y= data['y'][:]  # labels
y=np.transpose(y)
label= y[:,0]

print('The number of samples is ',subject_num,'subjects x',order_num,'model orders')
print('with dimension of',img_l,'x',img_h,'x',img_w)

############################################################################
##-----------------------train and test CNN model ------------------------##
############################################################################
# initialize hyperparameters
regularizer_param = 0.01     # regularization parameter for CNN weights
nb_epoch = 50                # epochs
bs = 64                      # batchsize
# Please note that we tuned regularization parameter,
# learning rate, and batchsize separately 
# for each cross-validation split based on the validation set.

# five-fold cross-validation 
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(5,random_state=2018)   # we used in the paper

for i,(train_index, test_index) in enumerate(skf.split(label,label)):    
    # construct training and validation set
    x_tmp = x[train_index]
    y_tmp = y[train_index]
    x_tmp = sklearn.utils.shuffle(x_tmp, random_state = 2018)
    y_tmp = sklearn.utils.shuffle(y_tmp, random_state = 2018)    
    x_train = x_tmp[:-int(train_index.shape[0] / 4)]
    y_train = y_tmp[:-int(train_index.shape[0] / 4)]
    x_val = x_tmp[-int(train_index.shape[0] / 4):]
    y_val = y_tmp[-int(train_index.shape[0] / 4):]    
        
    x_train = x_train.reshape([x_train.shape[0]*order_num,img_l,img_h,img_w])
    x_train = x_train.reshape(x_train.shape + (1,))    
    x_val = x_val.reshape([x_val.shape[0]*order_num,img_l,img_h,img_w])
    x_val = x_val.reshape(x_val.shape + (1,))    
    y_train = y_train.reshape([y_train.shape[0]*order_num,])
    y_train = to_categorical(y_train)    
    y_val = y_val.reshape([y_val.shape[0]*order_num,])
    y_val = to_categorical(y_val)       
    
    # construct the 3D-CNN
    model = Sequential()
    model.add(Conv3D(8, (3, 3, 3) , input_shape=(53, 63, 46, 1) , data_format="channels_last", kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer_param),name="Conv_1_1"))
    model.add(Activation('tanh'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, name="Pool_1"))    
    model.add(Conv3D(16, (3, 3, 3),kernel_initializer='lecun_uniform', kernel_regularizer=regularizers.l2(regularizer_param),name="Conv_1_2"))
    model.add(Activation('tanh'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, name="Pool_2"))
    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(regularizer_param),name="FC_1", activation='softmax'))
    model.add(BatchNormalization())
    model.add(Dense(2, kernel_initializer='lecun_uniform',kernel_regularizer=regularizers.l2(regularizer_param),name="FC_2", activation='softmax'))    
#    Adam1=optimizers.adam(lr=0.0001)  # change learning rate
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # optimizer = Adam1, if the learning rate is changed
    
    # train model and save the best result
    filepath ="./model_SSP_fold" + str(i) + "_weights.h5"  # set path
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1, save_best_only=True)
    hist = model.fit(x_train,y_train, validation_data=[x_val, y_val], batch_size=bs, epochs=nb_epoch, callbacks=[checkpoint]) 
    logpath = "./model_SSP_fold" + str(i) + "_log.pkl"
    pickle.dump(hist, open(logpath, 'wb'))
    
# classification performances
from sklearn.metrics import confusion_matrix   
from keras.models import load_model 
acc_v=[];sen_v=[];spec_v=[]     
acc_t=[];sen_t=[];spec_t=[]     

for i,(train_index, test_index) in enumerate(skf.split(label,label)):          
    x_tmp = x[train_index]
    y_tmp = y[train_index]
    x_tmp = sklearn.utils.shuffle(x_tmp, random_state = 2018)
    y_tmp = sklearn.utils.shuffle(y_tmp, random_state = 2018)    
    x_val = x_tmp[-int(train_index.shape[0] / 4):]
    y_val = y_tmp[-int(train_index.shape[0] / 4):]             
    x_val = x_val.reshape([x_val.shape[0]*order_num,img_l,img_h,img_w])
    x_val = x_val.reshape(x_val.shape + (1,))     
    y_val = y_val.reshape([y_val.shape[0]*order_num,])    
    x_test = x[test_index]
    y_test = y[test_index]
    x_test = x_test.reshape([x_test.shape[0]*order_num,img_l,img_h,img_w])
    x_test = x_test.reshape(x_test.shape + (1,))    
    y_test = y_test.reshape([y_test.shape[0]*order_num,])
    
    filepath ="./model_SSP_fold" + str(i) + "_weights.h5" 
    model = load_model(filepath) 
    
    # performance on validation set
    result_v=model.predict(x_val)
    y_pre_v = []
    for i in range(result_v.shape[0]):
        res = result_v[i,:].tolist()
        y_pre_v.append(res.index(max(res)))
    TN_v, FP_v, FN_v, TP_v = confusion_matrix(y_val, y_pre_v).ravel()    
    acc_v.append((TP_v+TN_v)/(TP_v+FP_v+FN_v+TN_v))
    sen_v.append(TP_v/(TP_v+FN_v))
    spec_v.append(TN_v/(FP_v+TN_v))
    
    # performance on test set
    result_t=model.predict(x_test)
    y_pre_t = []
    for i in range(result_t.shape[0]):
        res = result_t[i,:].tolist()
        y_pre_t.append(res.index(max(res)))
    TN_t, FP_t, FN_t, TP_t = confusion_matrix(y_test, y_pre_t).ravel()    
    acc_t.append((TP_t+TN_t)/(TP_t+FP_t+FN_t+TN_t))
    sen_t.append(TP_t/(TP_t+FN_t))
    spec_t.append(TN_t/(FP_t+TN_t)) 

for fold in range(skf.n_splits):
    print("Fold"+str(fold)+":")
    print("  ACC(validation): " + str(round(np.mean(acc_v[fold]),4)))
    print("  SEN(validation): " + str(round(np.mean(sen_v[fold]),4)))
    print("  SPEC(validation): " + str(round(np.mean(spec_v[fold]),4)))
    print("  ACC(test): " + str(round(np.mean(acc_t[fold]),4)))
    print("  SEN(test): " + str(round(np.mean(sen_t[fold]),4)))
    print("  SPEC(test): " + str(round(np.mean(spec_t[fold]),4)))
         