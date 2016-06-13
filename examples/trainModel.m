%% Keypoints detection in Human Pose Estimation
% This example demonstrates how to train and test a ConvNet to detect body
% keypoints from a single image. It is assumed that the individual is roughly
% localized.
%
% There is a demo code at *examples/testModel.m* for testing a
% pretrained model for 2D human pose estimation. In addition, there are a
% few testing examples at the folder *examples/img*. To run the demo,
% execute the scrpit *examples/testModel.m*.
%
% To train a model, the main execution scrpit is *examples/trainModel.m*.
% One should first download the traind and validation data (link will be
% provided soon) and then compile MatConvNet. MatConvNet is already included
% in this project. Instructions for compiling the library can be found 
% <http://www.vlfeat.org/matconvnet/install/#compiling here>. It is highly
% recommended to compile the library with the support of GPU (cuda and cudnn).
% For example, compiling the library with the support of the aforementioned
% functionalities can be done with the command:
%
% vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', 'EnableImreadJpeg', true, 'cudaRoot', '/usr/local/cuda-7.5', 'enableCudnn', true, 'cudnnRoot', '/home/vb/code/cudnn-v4'); 
%
% To run the command, you should incidate the filepath of *cudaRoot* and
% *cudnnRoot*. Moreover, the function *vl_compilenn()* is inside the folder *matlab*.
%
%% Dataset
% Training and validation data will be available soon. 
%
% Check the structure of the traind and validation files (*dataset/Train.mat* 
% and *dataset/Validation.mat*) to see the ground-truth format.
% The *2D coordinates* and the keypoints *visibility* are provided for each sample.
% When there is ground-truth for multiple individuals, it is store in the
% third dimension (e.g. ground-truth for 3 individuals will be stores in a
% 16x3x3 tensor).
%
% Storage files:
%
% # After downloading the train and validation files, update the
% *opts.DataMatTrain* and *opts.DataMatVal* variables with the files (e.g.
% lsp_dataset/Train.mat and lsp_dataset/Validation.mat).
% # When executing the main script, the train and validation files will be loaded to generate
% the image database. Specify where to store the image database in
% *opts.imdbPath* in advance (the default path is the *lsp_dataset* folder).
% # The directory to export the learned model should be provided in
% *opts.expDir* (the default path is the *lsp_dataset/model* folder).
% Notice that a learned model is stored at the end of every
% epoch. This means that restarting the training from scratch requires to
% empty the *opts.expDir* folder from previously store files.
%
% When the data is dowloaded and preprocessed, training a model can start by
% executing *examples/trainModel.m*.
%
% The rest of the documentation describes the main training script
% (*examples/trainModel.m*.).

%% Setup matconvnet
clearvars; close all; clc;
addpath('utils');
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

%% Storage Directories

% Train / Validation files (downloaded files)
opts.DataMatTrain='dataset/Train.mat';
opts.DataMatVal='dataset/Validation.mat';

% Export folder
opts.expDir = 'dataset/model';

% Image database file (it is generated at the beginning of the training)
opts.imdbPath = 'dataset/imdb.mat';

%% Parameters (adjustable)
% This parameters can be changed, although it is not recommended for this example. 

% Dataset name (used only for naming - not important)
opts.datas='Key'; 

% Input to the network (different from input image)
opts.patchHei=248; 
opts.patchWi=248;

% Imporant parameter for data augmentation (flip), extra clarification follows next.
opts.flipFlg='mpi';

% Batch size (should change only in case of memory problems)
opts.batchSize = 20; 

% Compensates for small batch size(if batchSize=10, then set to 2)
opts.numSubBatches = 1;

% Number of training epochs (after 30 epochs, it could be stopped)
opts.numEpochs = 30 ; 

% Learning rate (e.g. 20 epochs 10^-5, then 30 epochs 10^-6 - it is not recommended to change it)
opts.learningRate = [0.00001*ones(1, 20) 0.000001*ones(1, 30)] ;

%GPU index (if empy, CPU training)
opts.gpus = [1];

%% Parameters (fixed)
opts.useBnorm = false;
opts.bord=[];
opts.NoAug=1; %used for calling the proper imdb creation function
opts.prefetch = false ;
opts.outNode=16;%if heatmaps loss, then number of heatmaps
opts.inNode=3;
opts.lossFunc='l2loss-heatmap';
opts.batchNormalization = 1;%useful for big networks

%% Build the network
% The network is build separately.
% It's imporant to specify correctly the number of heatmaps (i.e.
% keypoints). This is done with *opts.outNode*.
% In this example, it is fixed to 16 heatmaps - keipoints.
net = initializeSpatialFusionNetwork(opts);

%% More parameters (fixed)

% Objectives for backpropagation
opts.derOutputs = {'objective', 1,'objective2', 1};

% Transformation to go from the input image to the heatmap space (only scale transformation, not to be changed) 
trf=[0.25 0 0 ; 0 0.25 0; 0 0 1];

% Parameters for the train scrpit
opts.net=net;
opts.numThreads = 15;
opts.transformation = 'f25' ;
opts.averageImage = single(repmat(128,1,1,opts.inNode));
opts.fast = 1;
opts.imageSize = [248, 248] ;
opts.border = [8, 8] ;
opts.bord=[0,0,0,0];

% Heatmap settings
opts.heatmap=1;
opts.bodyHeatmap=0; %body heatmap
opts.trf=trf;
opts.sigma=1.3;
opts.HeatMapSize=[62, 62];
opts.rotate=1;%rotation flag
opts.scale=1;%scale augm.

% Occluded keypoints
opts.inOcclud=1;

% Multiple instances
opts.multipInst=1;

% Heatmap scheme
opts.HeatMapScheme=1;

opts.train.momentum=0.95;

opts.negHeat=0;%set to 1 to include negative values for the occlusion
opts.ignoreOcc=1;%set to 1 to include negative values for the occlusion
opts.ignoreRest=1; %quasi single human training

opts.magnif=12;%amplifier
opts.facX=0.15;%pairwise heatmap width (def. 0.15)
opts.facY=0.08;%pairwise heatmap height

%% Heatmap Regressor
% After defining all parameters, build the image database and start training.
cnn_regressor_dag(opts);

%% Different Example
% It is possible to train a different example by:
%
% # Keeping the network architecture the same. A different architecture
% would require additional changes in the current script.
% # The file *flipKeyPointsCoords.m* has to be updated. This file includes
% the mapping of the keypoints in case of flip augmentation.
% In addition, the flip flag *opts.flipFlg* has to be
% updated with resepct to the new addition in the *flipKeyPointsCoords.m*.

%% Contact and Support
% For further questions and support, please contact
% Vasileios Belagiannis (vb@robots.ox.ac.uk).