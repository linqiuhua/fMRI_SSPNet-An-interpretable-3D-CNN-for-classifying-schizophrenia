"SSPNet: An interpretable CNN for classification of schizophrenia using phase maps of resting-state complex-valued fMRI data," published in Medical Image Analysis

   Requirements:
         Python 3.6.13   Keras 2.4.3

   Datatsets:
        Our results data can be downloaded from http://pan.dlut.edu.cn/share?id=cag3ass9t7ec
        The results data are obtained by using entropy bound minimization (EBM) algorithm of ICA (number of components N = 120, 10 runs, 82 subjects), and provided in data\res_SM_TC\res_SM_TC_k_*.mat. 
        We also provide the index for in-brain voxels in data\mask_ind.mat, and the spatial reference for DMN [1] and Auditory [2] in data\ref_DMN.mat and data\ref_AUD.mat.

   Experimental steps:    
   
        (1) SSPNet_trainingCode.py to train a 3D-CNN model using sample set of DMN spatial source phase separated from complex-valued fMRI data. 
        Note: The three models trained by spatial source phase, spatial source magnitude and magnitude-only data are stored in model\SSPNet_weights.h5, SSMNet_weights.h5 and MAGNet_weights.h5.

        (2) Run data_validation.m to generate bootstrap samples and resampling samples for intra- and inter- difference.
        Note: Our results of resampling samples for DMN and auditory cortex are stored in data\bs_DMN.mat, data\bs_AUD.mat, data\intra_inter_DMN.mat and data\intra_inter_AUD.mat.

        (3) Run main.m to perform Variance difference analysis and validation. 
        Note: Results for DMN and auditory cortex in the manuscript "Spatial Source Phase: A New Feature for Identifying Spatial Differences Based on Complex-Valued Resting-State fMRI" (submitted to Human Brain Mapping) can be obtained from demo_for_DMN.m and demo_for_Auditory.m.
