function net = initializeSpatialFusionNetwork(opts, varargin)

%Implmentation of SpatialNet with Fusion from:
%Pfister T. et al, Flowing ConvNets for Human Pose Estimation in Videos
%ICCV 2015, http://arxiv.org/abs/1506.02897.

% %Fine-tune a pretrained network
% if exist(varargin{1},'file')>=2
%     load(varargin{1});
%     net = dagnn.DagNN.loadobj(net);
%     net.move('cpu');
%     return;
% end

scal = 1 ;
init_bias = 0.0;
net.layers = {} ;
opts.cudnnWorkspaceLimit = 1024*1024*1024*5 ; % 5GB
opts.batchNormalization=0;
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;

% Conv 1
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(5,5, opts.inNode, 128, 'single'), ...%3X3, 32
    'biases', init_bias*ones(128,1, 'single'), ...
    'stride', 1, ... %1
    'pad', 2, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(128,1, 'single'), ...
      zeros(128,1, 'single'), zeros(128,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 1
net.layers{end+1} = struct('type', 'relu') ;

% Pool 1
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Conv 2
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(5,5, 128, 128, 'single'), ...%64
    'biases', init_bias*ones(128,1, 'single'), ...
    'stride', 1, ...
    'pad', 2, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(128,1, 'single'), ...
      zeros(128,1, 'single'), zeros(128,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 2
net.layers{end+1} = struct('type', 'relu') ;

% Pool 2
net.layers{end+1} = struct('type', 'pool', ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0) ;

% Conv 3
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(5,5, 128, 128, 'single'), ...
    'biases', init_bias*ones(128,1, 'single'), ...
    'stride', 1, ...
    'pad', 2, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(128,1, 'single'), ...
      zeros(128,1, 'single'), zeros(128,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 3
net.layers{end+1} = struct('type', 'relu') ;

% Conv 4
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(9,9, 128, 256, 'single'), ...
    'biases', init_bias*ones(256,1, 'single'), ...
    'stride', 1, ...%1
    'pad', 4, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(256,1, 'single'), ...
      zeros(256,1, 'single'), zeros(256,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 4
net.layers{end+1} = struct('type', 'relu') ;

% Conv 5
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(9,9, 256, 512, 'single'), ...
    'biases', init_bias*ones(512,1, 'single'), ...
    'stride', 1, ...%1
    'pad', 4, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(512,1, 'single'), ...
      zeros(512,1, 'single'), zeros(512,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 5
net.layers{end+1} = struct('type', 'relu') ;

% Conv 6
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(1,1,512,256,'single'),...
    'biases', init_bias*ones(256,1,'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(256,1, 'single'), ...
      zeros(256,1, 'single'), zeros(256,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 6
net.layers{end+1} = struct('type', 'relu') ;

% Conv 7
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(1,1,256,256,'single'),...
    'biases', init_bias*ones(256,1,'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(256,1, 'single'), ...
      zeros(256,1, 'single'), zeros(256,2, 'single')}}, 'learningRate', [2 1 0.3], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 7
net.layers{end+1} = struct('type', 'relu') ;

% Conv 8
net.layers{end+1} = struct('type', 'conv', ...
    'filters', 0.01/scal * randn(1,1,256,opts.outNode,'single'), ...
    'biases', init_bias*ones(opts.outNode,1, 'single'), ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 2], ...
    'weightDecay', [1 0], ...
    'opts', {convOpts});

if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm','weights', {{ones(opts.outNode,1, 'single'), ...
      zeros(opts.outNode,1, 'single'), zeros(opts.outNode,2, 'single')}}, 'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end

% ReLu 8
net.layers{end+1} = struct('type', 'relu') ;

% Loss-1
net.layers{end+1} = struct('type',opts.lossFunc) ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%covnert simpleNN to dagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
net.addLayer('error', dagnn.RegLoss('loss', 'mse-heatmap'), ...
    {'prediction','label'}, 'error') ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Add fusion layers
iter=1;

%concatenate conv3 and conv7 (skip connection)
net.addLayer('concat_f', dagnn.Concat(), {'x8','x16'}, {'x18'});

%conv1_f + ReLu
convBlock = dagnn.Conv('size', [7 7 384 64], 'pad', [3,3,3,3],'stride', [1,1], ...
    'hasBias', true);
net.addLayer('conv1_f', convBlock, {'x18'}, {'x19'}, {'conv1_f_filters', 'conv1_f_biases'}) ;

f = net.getParamIndex('conv1_f_filters') ;
net.params(f).value = 0.01.*randn([7 7 384 64], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('conv1_f_biases') ;
net.params(f).value = init_bias*ones(1, 64, 'single');

if opts.batchNormalization
    iter=iter+1;
    [net, ~] = addBnorm(net,'conv1_f',[7 7 384 64],'conv1_f',net.layers(end),iter,1);
end

reluBlock = dagnn.ReLU() ;
net.addLayer('relu1_f', reluBlock, {'x19'}, {'x20'}, {}) ;
%conv1_f + ReLu


%conv2_f + ReLu
convBlock = dagnn.Conv('size', [13 13 64 64], 'pad', [6,6,6,6],'stride', [1,1], ...
    'hasBias', true);
net.addLayer('conv2_f', convBlock, {'x20'}, {'x21'}, {'conv2_f_filters', 'conv2_f_biases'}) ;

f = net.getParamIndex('conv2_f_filters') ;
net.params(f).value = 0.01.*randn([13 13 64 64], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('conv2_f_biases') ;
net.params(f).value = init_bias*ones(1, 64, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;

if opts.batchNormalization
    iter=iter+1;
    [net, ~] = addBnorm(net,'conv1_f',[13 13 64 64],'conv2_f',net.layers(end),iter,1);
end

reluBlock = dagnn.ReLU() ;
net.addLayer('relu2_f', reluBlock, {'x21'}, {'x22'}, {}) ;
%conv2_f + ReLu

%conv3_f + ReLu
convBlock = dagnn.Conv('size', [13 13 64 128], 'pad', [6,6,6,6],'stride', [1,1], ...
    'hasBias', true);
net.addLayer('conv3_f', convBlock, {'x22'}, {'x23'}, {'conv3_f_filters', 'conv3_f_biases'}) ;

f = net.getParamIndex('conv3_f_filters') ;
net.params(f).value = 0.01.*randn([13 13 64 128], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('conv3_f_biases') ;
net.params(f).value = init_bias*ones(1, 128, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;

if opts.batchNormalization
    iter=iter+1;
    [net, ~] = addBnorm(net,'conv1_f',[13 13 64 128],'conv3_f',net.layers(end),iter,1);
end

reluBlock = dagnn.ReLU() ;
net.addLayer('relu3_f', reluBlock, {'x23'}, {'x24'}, {}) ;
%conv3_f + ReLu

%conv4_f + ReLu
convBlock = dagnn.Conv('size', [1 1 128 256], 'pad', [0,0,0,0],'stride', [1,1], ...
    'hasBias', true);
net.addLayer('conv4_f', convBlock, {'x24'}, {'x25'}, {'conv4_f_filters', 'conv4_f_biases'}) ;

f = net.getParamIndex('conv4_f_filters') ;
net.params(f).value = 0.01.*randn([1 1 128 256], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('conv4_f_biases') ;
net.params(f).value = init_bias*ones(1, 256, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;

reluBlock = dagnn.ReLU() ;
net.addLayer('relu4_f', reluBlock, {'x25'}, {'x26'}, {}) ;
%conv4_f + ReLu

%conv5_f + ReLu
convBlock = dagnn.Conv('size', [1 1 256 opts.outNode], 'pad', [0,0,0,0],'stride', [1,1], ...
    'hasBias', true);
net.addLayer('conv5_f', convBlock, {'x26'}, {'x27'}, {'conv5_f_filters', 'conv5_f_biases'}) ;

f = net.getParamIndex('conv5_f_filters') ;
net.params(f).value = 0.01.*randn([1 1 256 opts.outNode], 'single') ;
net.params(f).learningRate=1;
net.params(f).weightDecay=1;

f = net.getParamIndex('conv5_f_biases') ;
net.params(f).value = init_bias*ones(1, opts.outNode, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;

reluBlock = dagnn.ReLU() ;
net.addLayer('relu5_f', reluBlock, {'x27'}, {'prediction2'}, {}) ;
%conv5_f + ReLu

%objective
net.addLayer('objectiveFuse', dagnn.RegLoss('loss', opts.lossFunc), ...
    {'prediction2','label'}, 'objective2') ;

net.addLayer('error2', dagnn.RegLoss('loss', 'mse-heatmap'), ...
    {'prediction2','label'}, 'error2') ;

end

function [net, layerInput] = addBnorm(net,layerInput,dims,layerName,iterIdx,elemIdx)

x = net.getLayerIndex(layerInput);
inVar = net.layers(x).outputs;

params={sprintf('bn_%s_%d_%d_m',layerName,iterIdx,elemIdx),sprintf('bn_%s_%d_%d_b',layerName,iterIdx,elemIdx),sprintf('bn_%s_%d_%d_x',layerName,iterIdx,elemIdx)};
net.addLayer(sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx), dagnn.BatchNorm(), {inVar{1}}, {sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx)},params) ;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_m',layerName,iterIdx,elemIdx));
net.params(f).value = ones(dims(4), 1, 'single');
net.params(f).learningRate=2;
net.params(f).weightDecay=0;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_b',layerName,iterIdx,elemIdx));
net.params(f).value = ones(dims(4), 1, 'single');
net.params(f).learningRate=1;
net.params(f).weightDecay=0;
f = net.getParamIndex(sprintf('bn_%s_%d_%d_x',layerName,iterIdx,elemIdx));
net.params(f).value = ones(dims(4), 2, 'single');
net.params(f).learningRate=0.05;
net.params(f).weightDecay=0;

layerInput=sprintf('bn_%s_%d_%d',layerName,iterIdx,elemIdx);

end