function testModel()
% To test the pretrained model choose one image from the img folder.

%compile matconvet (CPU - only for demo)
if ~exist('../matlab/mex', 'dir')
    disp('Compilation please wait..')
    cd('../matlab');
    vl_compilenn();
    cd('../examples');
end

run(fullfile(fileparts(mfilename('fullpath')),...
    '..', 'matlab', 'vl_setupnn.m')) ;
addpath('utils');


% Model path (specify the folder of the trained model)
model_path='pre-trained';

% Model index (specify the index of the trained model)
ep = 30;

% Fixed parameters
opts.trf=[0.25 0 0 ; 0 0.25 0; 0 0 1];
loss={'prediction2'};
GPUt=[]; %cpu testing only - leave it empty
opts.imageSize = [248, 248];
opts.imageSizeInput = [256, 256];

GPUon=0;
if numel(GPUt)>0
    GPUon=1;
end

% Load model
load(sprintf('%s/net-epoch-%d.mat',model_path,ep));
net = dagnn.DagNN.loadobj(net) ;

if GPUon
    gpuDevice(GPUt);
    net.move('gpu');
else
    net.move('cpu');
end

%load an image
dy(1) = (opts.imageSizeInput(1) - opts.imageSize(1))/2;
dx(1) = (opts.imageSizeInput(2) - opts.imageSize(2))/2;

while 1
    
    [FileName,PathName] = uigetfile('*.jpg','Select a jpg file');
    img  = imread(fullfile(PathName,FileName));
    
    %pad the image
    diff = round((size(img,1)-size(img,2))/2);
    if diff>0 %pad width
        imt  = padarray(img,[0,diff]);
    else
        imt  = padarray(img,[-diff,0]);
        
    end
    imt = imresize(imt, [256,256]);
    
    %crop input to the proper size
    imtCrop=imt(dy(1):dy(1)+opts.imageSize(2)-1,dx(1):dx(1)+opts.imageSize(1)-1,:);
    
    %single format and mean subtraction
    im_ = single(imtCrop);
    
    if GPUon
        im_ = gpuArray(im_);
    end
    
    im_ = bsxfun(@minus, im_, single(repmat(128,1,1,3))) ; %subtract mean
    im_ = im_./256;
    
    %evaluate the image
    net.mode='test';
    net.eval({'input', im_}) ;
    
    %gather the requested predictions
    output = cell(numel(loss,1));
    for i=1:numel(loss)
        output{i} = net.vars(net.getVarIndex(loss{i})).value ;
    end
    
    for j=1:size(output{1},3)
        %plot all pairs of heatmaps
        plotPairHeatMap(opts,j,dx,dy,imt,output,j);
    end
end

end

function plotPairHeatMap(opts,keyp,dx,dy,imt,output,ch)

labels  = {'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', ...
    'Left Ankle', 'Torso', 'Neck', 'Lower Head', 'Upper Head', 'Right Wirst', ...
    'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wirst'};
%coords
heat = output{1}(:,:,ch);
[y,x,~] = find( heat == max(heat(:)));
x = ((x(1)-1) * 256)/62;
y = ((y(1)-1) * 256)/62;


tform = affine2d(opts.trf');
imtCrop=imt(dy(1):dy(1)+opts.imageSize(2)-1,dx(1):dx(1)+opts.imageSize(1)-1,:);
heat_mapRe = imresize(output{1}(:,:,ch),1/opts.trf(1,1));
IMG_gray = uint8(heat_mapRe./(max(heat_mapRe(:)))*256);
map = colormap(jet(256)); % Get the current colormap
I = ind2rgb(IMG_gray,map);
Ires = uint8(150*I+0.8*double(imtCrop));
I=imtCrop;
subtightplot(2,1,1,0.05);imshow(I);title('Press Space to Continue'); %hold on; pause();
set(gca,'FontSize',15);
subtightplot(2,1,2,0.05);imshow(Ires);title(sprintf('%s',labels{keyp}));
text(5,5,sprintf('2D coordinates: %.1f %.1f',x,y),'Color','green','FontSize',14);
set(gca,'FontSize',15);
pause();

end