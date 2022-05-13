"SSPNet: An interpretable CNN for classification of schizophrenia using phase maps of resting-state complex-valued fMRI data," published in Medical Image Analysis

   Requirements:
         Python 3.6.13   Keras 2.4.3

   Datatsets:
        Our sample sets can be downloaded from https://pan.baidu.com/s/1mhppL7V9rd224XLwIezdPQ (password: iin0).
        
        (1) SSPsamples.mat: sample set of DMN spatial source phase separated from complex-valued fMRI data.
        (2) SSMsamples.mat: sample set of DMN spatial source magnitude separated from complex-valued fMRI data.
        (3) MAGsamples.mat: sample set of DMN separated from magnitude-only fMRI data.

   Experimental steps:    
   
        (1) SSPNet_trainingCode.py to train a 3D-CNN model using sample set of DMN spatial source phase separated from complex-valued fMRI data. 
        Note: The three models trained by spatial source phase, spatial source magnitude and magnitude-only data are stored in model\SSPNet_weights.h5, SSMNet_weights.h5 and MAGNet_weights.h5.

        (2) main.py to perform classification and generate interpretability results (testing code).
