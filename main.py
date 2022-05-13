
# Submission for Medical Image Analysis 
# "SSPNet: An interpretable 3D-CNN 
# for classification of schizophrenia 
# using phase maps 
# of resting-state complex-valued fMRI data"

from keras import layers
from keras import backend as K
from keras.models import Model
from keras.models import load_model

import numpy as np 
import matplotlib.pyplot as plt 

import h5py
import scipy.io as sio 
from scipy import ndimage

import os
os.chdir('E:/NYW/code&data/')   # set path

############################################################################
##-------------------------define functions-------------------------------##
############################################################################
# obtain feature maps at the 1st convolutional layer
def Conv1(sam,model):
    conv_layer = model.get_layer('Conv_1_1') 
    iterate = K.function([model.input], [conv_layer.output[0]])   
    conv_layer_output_value = iterate([sam])    
    featuremap = np.mean(conv_layer_output_value, axis=4) # averaging
    featuremap = np.reshape(featuremap,[featuremap.shape[1],featuremap.shape[2],featuremap.shape[3]])
    return featuremap

# obtain feature maps at the 2nd convolutional layer
def Conv2(sam,model):
    conv_layer = model.get_layer('Conv_1_2') 
    iterate = K.function([model.input], [conv_layer.output[0]])   
    conv_layer_output_value = iterate([sam])    
    featuremap = np.mean(conv_layer_output_value, axis=4) 
    featuremap = np.reshape(featuremap,[featuremap.shape[1],featuremap.shape[2],featuremap.shape[3]])    
    return featuremap

# generate Grad-CAM heatmaps
def Grad_cam(sam,model):   
    preds_class=np.argmax(model.predict(sam)[0])
    pred_result = model.output[:,preds_class]   
    last_pool_layer = model.get_layer('Pool_2') 
    grads = K.gradients(pred_result, last_pool_layer.output)[0] 
    pooled_grads = K.mean(grads, axis=(0, 1, 2, 3))    
    iterate = K.function([model.input], [pooled_grads, last_pool_layer.output[0]])      
    pooled_grads_value, pool_output_value = iterate([sam])    
    for i in range(16):     
        pool_output_value[:, :, :, i] *= pooled_grads_value[i]          
    heatmap = np.mean(pool_output_value, axis=-1)     
    heatmap = np.maximum(heatmap, 0) 
    heatmap /= np.max(heatmap)   
    return heatmap                  

# show results of 9 slices
def show(maps):
    tmp = np.zeros((63,53,9))
    fig = plt.figure(figsize = (5.5, 5.5))
    for i in range(9):
        tmp[:,:,i] = np.flip(np.swapaxes(maps[:,:,23+i],0,1), 0);
        ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
        ax.set_title('Slice'+str(i+23),size = 12)    
        plt.imshow(tmp[:,:,i], cmap = 'jet')
        plt.colorbar()
    return plt

############################################################################
##-----------------------------load data----------------------------------##
############################################################################
data_ssp = h5py.File('SSPsamples.mat')
data_ssm = h5py.File('SSMsamples.mat')
data_mag = h5py.File('MAGsamples.mat')

x_ssp = np.transpose(data_ssp['x'][:])  # samples
y_ssp = np.transpose(data_ssp['y'][:])  # labels
x_ssm = np.transpose(data_ssm['x'][:])  
y_ssm = np.transpose(data_ssm['y'][:]) 
x_mag = np.transpose(data_mag['x'][:])  
y_mag = np.transpose(data_mag['y'][:]) 

subject_num = x_ssp.shape[0] # num of subject
order_num = x_ssp.shape[4]   # num of model order
img_l = x_ssp.shape[1]       # length of sample
img_h = x_ssp.shape[2]       # height of sample
img_w = x_ssp.shape[3]       # width of sample

print('The number of samples is: ',subject_num,'subjects x',order_num,'model orders')
print('The size of samples is: ',img_l,'x',img_h,'x',img_w)

###########################################################################
##--------- test classification performance of CNN model-----------------##
###########################################################################
modelpath1 = "./model/SSPNet_weights.h5" # load model
modelpath2 = "./model/SSMNet_weights.h5"
modelpath3 = "./model/MAGNet_weights.h5"
model_ssp = load_model(modelpath1)
model_ssm = load_model(modelpath2)
model_mag = load_model(modelpath3)
model_ssp.summary()                      # show details of the model

# select the test sample
# We selected the 7th and 65th subjects to generate the individual results 
sub = 7  # subject
mo = 10  # model order 

# model predict
# ------- SSP ---------#
sam_ssp = x_ssp[sub,:,:,:,mo]
lab_ssp = y_ssp[sub,mo]
sam_ssp = np.expand_dims(sam_ssp, axis = 0)
sam_ssp = np.expand_dims(sam_ssp, axis = 4)
pre_ssp = model_ssp.predict(sam_ssp)[0,:].tolist().index(max(model_ssp.predict(sam_ssp)[0,:].tolist()))

# show the predicted result of the sample
if lab_ssp == 0:
    print('The sample of SSP is from an HC.')
    if pre_ssp == lab_ssp:
        print('The prediction of SSPNet is correct.') 
    if pre_ssp != lab_ssp:
        print('The prediction of SSPNet is wrong.')
if lab_ssp == 1:
    print('The sample of SSP is from an SZ.')
    if pre_ssp == lab_ssp:
        print('The prediction of SSPNet is correct.') 
    if pre_ssp != lab_ssp:
        print('The prediction of SSPNet is wrong.')

# ------- SSM ---------#
sam_ssm = x_ssm[sub,:,:,:,mo]
lab_ssm = y_ssm[sub,mo]
sam_ssm = np.expand_dims(sam_ssm, axis = 0)
sam_ssm = np.expand_dims(sam_ssm, axis = 4)
pre_ssm = model_ssm.predict(sam_ssm)[0,:].tolist().index(max(model_ssm.predict(sam_ssm)[0,:].tolist()))

if lab_ssm == 0:
    print('The sample of SSM is from an HC.')
    if pre_ssm == lab_ssm:
        print('The prediction of SSMNet is correct.') 
    if pre_ssm != lab_ssm:
        print('The prediction of SSMNet is wrong.')
if lab_ssm == 1:
    print('The sample of SSM is from an SZ.')
    if pre_ssm == lab_ssm:
        print('The prediction of SSMNet is correct') 
    if pre_ssm != lab_ssm:
        print('The prediction of SSMNet is wrong.')

# ------- MAG ---------#
sam_mag = x_mag[sub,:,:,:,mo]
lab_mag = y_mag[sub,mo]
sam_mag = np.expand_dims(sam_mag, axis = 0)
sam_mag = np.expand_dims(sam_mag, axis = 4)
pre_mag = model_mag.predict(sam_mag)[0,:].tolist().index(max(model_mag.predict(sam_mag)[0,:].tolist()))

if lab_mag == 0:
    print('The sample of MAG is from an HC.')
    if pre_mag == lab_mag:
        print('The prediction of MAGNet is correct.') 
    if pre_mag != lab_mag:
        print('The prediction of MAGNet is wrong.')
if lab_mag == 1:
    print('The sample of MAG is from an SZ.')
    if pre_mag == lab_mag:
        print('The prediction of MAGNet is correct') 
    if pre_mag != lab_mag:
        print('The prediction of MAGNet is wrong.')

###########################################################################
##---------------generate interpretability results ----------------------##
###########################################################################     

# --------visualizations of the 1st convolutional layer-----------#
# SSP provided more intact features compared to SSM and MAG, and emphasized the DMN region edges.
        
# ------- SSP ---------#           
FM1_ssp = Conv1(sam_ssp,model_ssp)                          
uFM1_ssp = ndimage.zoom(FM1_ssp,[img_l/FM1_ssp.shape[0],img_h/FM1_ssp.shape[1],img_w/FM1_ssp.shape[2]],order=2)
                                            # upsample
show(uFM1_ssp)

# ------- SSM ---------#           
FM1_ssm = Conv1(sam_ssm,model_ssm)                                    
uFM1_ssm = ndimage.zoom(FM1_ssm,[img_l/FM1_ssm.shape[0],img_h/FM1_ssm.shape[1],img_w/FM1_ssm.shape[2]],order=2)                                          
show(uFM1_ssm)

# ------- MAG ---------#           
FM1_mag = Conv1(sam_mag,model_mag)                                  
uFM1_mag = ndimage.zoom(FM1_mag,[img_l/FM1_mag.shape[0],img_h/FM1_mag.shape[1],img_w/FM1_mag.shape[2]],order=2)                                          
show(uFM1_mag)

# --------visualizations of the 2nd convolutional layer-----------#
# SSP retained the intact features and emphasize the activations inside the DMN region.
# The features of SSM and MAG tended to be enhanced compared to those of the first convolutional layer. 

# ------- SSP ---------#           
FM2_ssp = Conv2(sam_ssp,model_ssp)                                  
uFM2_ssp = ndimage.zoom(FM2_ssp,[img_l/FM2_ssp.shape[0],img_h/FM2_ssp.shape[1],img_w/FM2_ssp.shape[2]],order=2)                                            
show(uFM2_ssp)

# ------- SSM ---------#                                            
FM2_ssm = Conv2(sam_ssm,model_ssm)                                   
uFM2_ssm = ndimage.zoom(FM2_ssm,[img_l/FM2_ssm.shape[0],img_h/FM2_ssm.shape[1],img_w/FM2_ssm.shape[2]],order=2)                                            
show(uFM2_ssm)
# ------- MAG ---------#           
FM2_mag = Conv2(sam_mag,model_mag)                                  
uFM2_mag = ndimage.zoom(FM2_mag,[img_l/FM2_mag.shape[0],img_h/FM2_mag.shape[1],img_w/FM2_mag.shape[2]],order=2)                                          
show(uFM2_mag)

# ------------------------heatmaps------------------------------#
# SSP localized DMN related regions with opposite strengths for HCs and SZs. 
# SSM localized PCC and IPL for both HCs and SZs,
# and MAG localized PCC for both HCs and SZs.

# ------- SSP ---------#           
GM_ssp = Grad_cam(sam_ssp,model_ssp)                                   
uGM_ssp = ndimage.zoom(GM_ssp,[img_l/GM_ssp.shape[0],img_h/GM_ssp.shape[1],img_w/GM_ssp.shape[2]],order=2)                                            
show(uGM_ssp)

# ------- SSM ---------#                                            
GM_ssm = Grad_cam(sam_ssm,model_ssm)                                   
uGM_ssm = ndimage.zoom(GM_ssm,[img_l/GM_ssm.shape[0],img_h/GM_ssm.shape[1],img_w/GM_ssm.shape[2]],order=2)                                            
show(uGM_ssm)

# ------- MAG ---------#           
GM_mag = Grad_cam(sam_mag,model_mag)                                  
uGM_mag = ndimage.zoom(GM_mag,[img_l/GM_mag.shape[0],img_h/GM_mag.shape[1],img_w/GM_mag.shape[2]],order=2)                                          
show(uGM_mag)
