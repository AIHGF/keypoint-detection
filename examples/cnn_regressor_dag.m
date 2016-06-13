function [net, info] = cnn_regressor_dag(varargin)
% Deep regressor mainly for human pose estimation
 
%Dataset
opts.datas='BBC';

%LSP params for augmentation
opts.patchHei=120;
opts.patchWi=80;

%Camera
opts.cam=1;

%augmentation
opts.aug=0;
opts.NoAug=0;

opts.expDir = sprintf('/data/vb/Temp/%s-baseline%d',opts.datas,opts.cam) ;
opts.imdbPath = fullfile(opts.expDir, sprintf('imdb%d.mat',opts.cam));

opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 100 ;
opts.train.continue = true ;
opts.train.derOutputs= {'objective', 1} ;
opts.train.learningRate = [0.001*ones(1, 17) 0.0005*ones(1, 50) 0.002*ones(1, 500)  0.03*ones(1, 130) 0.01*ones(1, 100)] ;
opts.train.momentum=0.9;
opts.useBnorm = false ;
opts.batchNormalization = 0;
opts.train.prefetch = false ;

%GPU
opts.train.gpus = [];

%architecture parameters
opts.initNet=''; %pre-trained network
opts.outNode=14;%14 bbc, 18,28,42
opts.outPairNode=8;% pairwise terms
opts.outCombiNode=5;
opts.inNode=3;
opts.lossFunc='tukeyloss-heatmap';
opts.lossFunc2=[];
opts.lossFunc3=[];
opts.lossFunc4=[];
opts.errMetric =[];
opts.train.thrs=0;
opts.train.refine=false;
opts.HighRes = 0; %high resolution output
opts.ConcFeat=768;  %number of channels at concat
opts.skip_layer = ''; %skip layer
opts.train.hardNeg=0;%hard negative mining

%axis error plot (x,y)
opts.train.scbox=opts.patchWi*ones(opts.outNode,1);
opts.train.scbox(2:2:end)=opts.patchHei;

%cross validation
opts.cvset=[];
opts.cvidx=[];

opts.DataMatTrain=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsT%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);
opts.DataMatVal=sprintf('/mnt/ramdisk/vb/%s/%s_imdbsV%daug%d.mat',opts.datas,opts.datas,opts.cam,opts.aug);

%load network
opts.lossWeight=1;
net = [];

%get batch params
bopts.numThreads = 15;
bopts.transformation = 'f5' ;
bopts.averageImage = single(repmat(128,1,1,opts.inNode));%256/2
bopts.fast = 1;
bopts.imageSize = [120, 80] ;
bopts.border = [10, 10] ;
bopts.bord=[0,0,0,0]; %cropping border
bopts.heatmap=0;
bopts.bodyHeatmap=0;
bopts.trf=[];
bopts.sigma=[];
bopts.HeatMapSize=[];
bopts.flipFlg='bbc';%full, bbc
bopts.inOcclud=1; %include occluded points
bopts.multipInst=1; %include multiple instances in the heatmaps
bopts.HeatMapScheme=1; %how to generate heatmaps
bopts.HeatMapVal=100; %assigned value for the heatmap center
bopts.rotate=0;%rotation augm.
bopts.scale=0;%scale augm.
bopts.pairHeatmap=0;
bopts.bodyPairs = [];
bopts.negHeat=0;%set to 1 to include negative values for the occlusion
bopts.ignoreOcc=0;%requires 
bopts.magnif=8;%amplifier for the body heatmaps
bopts.facX=0.15;%pairwise heatmap width
bopts.facY=0.08;%pairwise heatmap height
bopts.combiHeatmap=0;
bopts.bodyCombis=[];
bopts.ignoreRest=0; %quasi single human training

%parse settings
[opts, trainParams] = vl_argparse(opts, varargin); %main settings
[opts.train, boptsParams]= vl_argparse(opts.train, trainParams); %train settings
[bopts, netParams]= vl_argparse(bopts, boptsParams); %batch settings
net=netParams{1}.net; %network
clear trainParams boptsParams netParams;

opts.train.bodyPairs = bopts.bodyPairs;%structured prediction training
opts.train.trf =  bopts.trf;%transformation from the input to the output space

useGpu = numel(opts.train.gpus) > 0 ;
bopts.GPU=useGpu;

%Paths OSX / Ubuntu
opts.train.expDir = opts.expDir ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
else
    if isempty(opts.cvset)
        if opts.aug==0 || opts.NoAug==1
            imdb = getImdbNoAug(opts) ; %normal training
        else
            %to be added
        end
    else
         %cross validation - not active anymore
    end
    
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end
         
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

fn = getBatchDagNNWrapper(bopts,useGpu) ;
  
info = cnn_train_dag(net, imdb, fn, opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------

[im, lab] = cnn_get_batch_keypoints(imdb, batch, opts, ...
                            'prefetch', nargout == 0) ;
if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', lab} ;
end

% --------------------------------------------------------------------
function imdb = getImdbNoAug(opts)
% --------------------------------------------------------------------

disp('Database generation. It will take some minutes...');

load(opts.DataMatTrain); %training data

imdb.images.data=imgPath;
sets=ones(1,numel(imgPath));
imdb.images.labels=ptsAll;

clear imgPath ptsAll;

load(opts.DataMatVal); %validation data

sets=[sets 2*ones(1,numel(imgPath))];
imdb.images.data=[imdb.images.data imgPath];

if iscell(imdb.images.labels)%different formats of ground-truth
    imdb.images.labels=[imdb.images.labels ptsAll];
else
    imdb.images.labels=cat(3,imdb.images.labels,ptsAll);
end

imdb.images.set=sets;
imdb.meta.sets = {'train', 'val', 'test'} ;

disp('Database generation done. Storing the database will take a minute...');