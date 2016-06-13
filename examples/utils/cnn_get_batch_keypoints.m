function [imo, labels] = cnn_get_batch_keypoints(imdb, batch, varargin)

opts.imageSize = [120, 80] ; %input image size
opts.border = [10, 10] ; %crop interval
opts.keepAspect = false ;
opts.numAugments = 1 ;
opts.transformation = 'f5' ;
opts.averageImage = [];
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 15 ;
opts.prefetch = false ;
opts.fast=false;
opts.bord = [];%cropping border
opts.heatmap=0;%include keypoint heatmaps
opts.bodyHeatmap=0;%include body heatmaps
opts.pairHeatmap=0; %include body part heatmap
opts.trf=[]; %transformation from input image to ouput heatmap
opts.sigma=[]; %for the keypoints' gaussian
opts.HeatMapSize=[]; %output heatmap
opts.flipFlg='bbc'; %flip policy
opts.inOcclud=1; %include occluded keypoints
opts.multipInst=1; %include multiple instances
opts.GPU=0; %activate / deactivate GPU
opts.HeatMapScheme=1; %how to generate heatmaps (keep it to 1)
opts.HeatMapVal=100; %keypoints' heatmap value
opts.rotate=0;%rotation augmentation
opts.scale=0;%scale augmentation
opts.negHeat=0; %allow negative heatmaps for occluded keypoints (requires opts.inOcclud=1)
opts.ignoreOcc=0; %flag for ignoring the gradients of occluded keypoints
opts.bodyPairs = []; %set of body pairs (manually defined in the config file)
opts.magnif=8; %magnification for body parts heatmap
opts.facX=0.15;%scale - pairwise heatmap width
opts.facY=0.08;%scale - pairwise heatmap height
opts.combiHeatmap=0; %combinations of body parts
opts.bodyCombis=[]; %TO BE REMOVED
opts.ignoreRest=0; %in case of multiple individuals, ignore the gradients of the rest indiv. (keep only for active indiv.)

opts = vl_argparse(opts, varargin);

%DATA FETCHING AND PRE-FETCHING
%It should be used only when the data doesn't fit into the RAM

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(imdb.images.data) >= 1 && ischar(imdb.images.data{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
    vl_imreadjpeg(imdb.images.data(batch), 'numThreads', opts.numThreads, 'prefetch') ;
    imo = [];
    return ; %stop here
end

if fetch
    im = vl_imreadjpeg(imdb.images.data(batch),'numThreads', opts.numThreads) ;
else
    im = imdb.images.data(batch) ;
end

%data augmentation: crop transformations
transformations = generateTran(opts,batch);

%redundant information - to be removed
if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
    opts.averageImage = zeros(1,1,3) ;
end

%redundant information - to be removed
if numel(opts.averageImage) == 3
    opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

%the actual data after preprocessing
imo = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
    numel(batch)*opts.numAugments, 'single') ;

%label format (stored in cells)
if opts.heatmap %to be changed
    %1 - keypoints, 2 - heatmap, 3 - weight mask, 4 - number of instances,
    %5 - body heatmap (removed), 6 - body weight mask (removed),
    %7 - segmentation mask (removed), 8 - pairwise heatmap,
    %9 - pairwise weight mask, 10 combination heatmap,
    %11 combination weight mask (not anymore)
    %12 occlusion binary map
    
    labels=cell(12,numel(batch));
else
    error('There is currently support only for the heatmaps') ;
end



for i=1:numel(batch)
    fr=batch(i); %frame / image number
    
    poseGT=[]; %keypoints for the main instacne
    poseRest=[]; %rest instances
    
    %get image and labels
    [img, poseGT,poseRest, poseOccl, poseRestOccl] = parseData(opts,imdb,fr,i, im);
    imt=img;%old structure name
    tempY=poseGT;%old structure name
    tempRest=poseRest;%old structure name
    
    %minimum number of individuals
    insta=1;
    
    %rotate augmentation
    if opts.rotate && rand(1)>0.5
        pts = [tempY;reshape(permute(tempRest,[1,3,2]),[],2)];
        [imt,pts] = imRotateGetBatch(imt,pts);
        
        tempY = pts(1:size(tempY,1),:);
        pts=pts(size(tempY,1)+1:end,:);
        cc=1;
        while size(pts,1)>0
            tempRest(:,:,cc) = pts(1:size(tempY,1),:);
            pts=pts(size(tempY,1)+1:end,:);
            cc=cc+1;
        end
        clear pts;
    end
    
    %scale augmentation
    if opts.scale && rand(1)>0.5
        pts = [tempY;reshape(permute(tempRest,[1,3,2]),[],2)];
        [imt,pts] = imScaleGetBatch(imt,pts);
        
        tempY = pts(1:size(tempY,1),:);
        pts=pts(size(tempY,1)+1:end,:);
        cc=1;
        while size(pts,1)>0
            tempRest(:,:,cc) = pts(1:size(tempY,1),:);
            pts=pts(size(tempY,1)+1:end,:);
            cc=cc+1;
        end
        clear pts;
    end
    
    %crop and flip augmenation
    w = size(imt,2) ;
    h = size(imt,1) ;
    
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
    switch opts.transformation
        case 'stretch'
            sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [w;h])) ;
            dx = randi(w - sz(2) + 1, 1) ;
            dy = randi(h - sz(1) + 1, 1) ;
            flip = rand > 0.5 ;
        otherwise
            %tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
            tf = tfs(:, transformations(1,i)) ;
            %tf=[0,0,0]';%debug
            sz = opts.imageSize(1:2) ;
            dx = floor((w - sz(2)) * tf(2)) + 1 ;
            dy = floor((h - sz(1)) * tf(1)) + 1 ;
            flip = tf(3) ;
    end
    
    %exclude missing annotation
    idx=tempY(:,1)>0 & tempY(:,2)>0; %zeros in x, means zeros in y as well
    
    %check if all keypoints are within image frame
    checkY(:,1)=tempY(idx,1)-dx+1;
    checkY(:,2)=tempY(idx,2)-dy+1;
    if sum(checkY(:)<0) ~=0
        ofsX=min(0,min(checkY(:,1)));
        ofsY=min(0,min(checkY(:,2)));
        ofsX=ofsX -(ofsX<0); %origin 1,1
        ofsY=ofsY -(ofsY<0); %origin 1,1
        dx = floor(dx+ofsX);
        dy = floor(dy+ofsY);
    end
    clear checkY;
    
    %updated dx,dy
    checkY(:,1)=tempY(idx,1)-dx+1;
    checkY(:,2)=tempY(idx,2)-dy+1;
    if sum(checkY(:,2)> (sz(1) + dy)) ~=0 || sum(checkY(:,1)> (sz(2) + dx)) ~=0
        ofsX=max(sz(2)+dx-1,max(checkY(:,1))) - (sz(2)+dx-1);
        ofsY=max(sz(1)+dy-1,max(checkY(:,2))) - (sz(1)+dy-1);
        dx = floor(dx+ofsX);
        dy = floor(dy+ofsY);
    end
    clear checkY;
    
    %crop keypoints
    tempY(idx,1)=tempY(idx,1)-dx+1;
    tempY(idx,2)=tempY(idx,2)-dy+1;
    
    clear idx;
    
    %crop images points
    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    
    if flip %main istance
        sx = fliplr(sx) ;
        tempY = flipKeyPointsCoords(opts.imageSize(2),tempY,opts.flipFlg); %flip keypoints
    end
    
    %crop extra points
    if sum(size(tempRest))>0
        insta=insta+size(tempRest,3); %number of individuals
        
        for j=1:size(tempRest,3)
            
            %exclude missing annotation
            idx= tempRest(:,1,j)>0 & tempRest(:,2,j)>0;
            
            tempRest(idx,1,j) = tempRest(idx,1,j) - dx +1;
            tempRest(idx,2,j) = tempRest(idx,2,j) - dy +1;
            
            clear idx;
            
            if flip %rest instnaces
                tempRest(:,:,j) = flipKeyPointsCoords(opts.imageSize(2), tempRest(:,:,j),opts.flipFlg);
            end
            
            %explude the out-of-plane points
            idx= (tempRest(:,1,j) > numel(sx)) | (tempRest(:,1,j) < 1); %X coord.
            tempRest(idx,:,j) = 0; %both x nad y
            idx= (tempRest(:,2,j) > numel(sy)) | (tempRest(:,2,j) < 1); %Y coord.
            tempRest(idx,:,j) = 0; %both x nad y
            
            clear idx;
        end
    end
    
    %store labels
    labels{1,i}=tempY;%single skeleton output for regression (not used)
    
    
    %Keypoint heatmap generations
    if opts.heatmap
        
        %heatmap size (defined based on the output of the network)
        heatmap=zeros(opts.HeatMapSize(1),opts.HeatMapSize(2),size(tempY,1));
        heatmap_mask=zeros(opts.HeatMapSize(1),opts.HeatMapSize(2),size(tempY,1));
        
        %first: rest of keypoints - heatmaps
        if sum(size(tempRest))>0
            poseRest = zeros(size(tempRest));
            for k=1:size(tempRest,3)
                %transform the keypoints to the output heatmap space
                restY=  (opts.trf*[tempRest(:,:,k) ones(size(tempRest(:,:,k),1),1)]')';
                poseRest(:,:,k)=restY(:,1:2); %abuse of poseRest, change this name
                
                %add first the rest keypoints and then the main
                for j=1:size(poseRest,1)
                    %fix rounding problems
                    if poseRest(j,1,k)>0 && poseRest(j,2,k)>0 %missing keypoints
                        x=min(max(1,round(poseRest(j,1,k))),size(heatmap,2));
                        y=min(max(1,round(poseRest(j,2,k))),size(heatmap,1));
                        
                        topts=opts;
                        if topts.ignoreOcc %negative values in order to ignore
                            topts.magnif=topts.magnif.*poseRestOccl(j,k); %occlusion map
                        end
                        if topts.ignoreRest && topts.magnif>0 %second constraint because of the above
                            topts.magnif=topts.magnif.*(-1); %single indiv. ignore rest
                        end
                        topts.facX=1;
                        topts.facY=1;
                        heatmap(:,:,j) = generateGHeatmap(heatmap(:,:,j),[x,y],180,topts.sigma,topts);
                    end
                end
            end
        end
        
        %second: main keypoints - heatmaps
        %transform the keypoints to the output heatmap space
        poseMAP = (opts.trf*[tempY ones(size(tempY,1),1)]')';
        poseMAP = poseMAP(:,1:2);
        
        for j=1:size(poseMAP,1)
            %fix rounding problems
            if poseMAP(j,1)>0 && poseMAP(j,2)>0
                x=min(max(1,round(poseMAP(j,1))),size(heatmap,2));
                y=min(max(1,round(poseMAP(j,2))),size(heatmap,1));
                
                topts=opts;
                if topts.ignoreOcc %negative values in order to ignore
                    topts.magnif=topts.magnif.*poseOccl(j); %occlusion map
                end
                topts.facX=1;
                topts.facY=1;
                heatmap(:,:,j) = generateGHeatmap(heatmap(:,:,j),[x,y],180,topts.sigma,topts);
            end
            
            %generate weights for balancing positive / negative cells
            heatmap_mask = getWeightMask(opts,insta,0,heatmap,j,heatmap_mask);
            
            %visualization
            %mapVisualize(opts, imt, sy, sx, heatmap, j, poseMAP, tempY, poseRest);
        end
        
        %store labels
        labels{2,i} = heatmap; %keypoint heatmap
        labels{3,i} = heatmap_mask; %keypoint weighting mask
    end
    
    %Generating heatmaps for combinations of keypoints
    if opts.pairHeatmap
        bodyPairs = opts.bodyPairs;
        
        %pair and combi - heatmaps
        pair_heatmap=zeros(opts.HeatMapSize(1),opts.HeatMapSize(2),size(opts.bodyPairs,2));
        pair_heatmap_mask=zeros(opts.HeatMapSize(1),opts.HeatMapSize(2),size(opts.bodyPairs,2));
        
        for j=1:size(bodyPairs,2)
            part_A = bodyPairs(1,j);
            part_B = bodyPairs(2,j);
            
            %go through the rest keypoints first
            if sum(size(tempRest))>0
                for k=1:size(tempRest,3)
                    if opts.ignoreOcc %in case of occluded keypoint, remove the part
                        if poseRestOccl(part_A,k)==-1
                            poseRest(part_A,:,k)=0;
                        end
                        if poseRestOccl(part_B,k)==-1
                            poseRest(part_B,:,k)=0;
                        end
                    end
                    
                    if sum(poseRest(part_A,:,k))>0 && sum(poseRest(part_B,:,k))>0 %both keypoints availiable
                        
                        %get mu, sigma and theta for generating the heatmaps
                        [part_center, theta, len] = getHeatMapParams(poseRest(part_A,:,k),poseRest(part_B,:,k));
                        
                        %in case of single individual training, ignore the rest parts
                        if opts.ignoreRest
                            opts.magnif=-opts.magnif;
                        end
                        
                        %generate the heatmap
                        pair_heatmap(:,:,j) = generateGHeatmap(pair_heatmap(:,:,j),part_center,theta,len,opts);
                        
                        %restore the magnif value to be positive
                        if opts.ignoreRest
                            opts.magnif=-opts.magnif;
                        end
                        
                    end
                end
            end
            
            if opts.ignoreOcc %in case of occluded keypoint, remove the part
                if poseOccl(part_A)==-1
                    poseMAP(part_A,:)=0;
                end
                if poseOccl(part_B)==-1
                    poseMAP(part_B,:)=0;
                end
            end
            
            %go through the main keypoints
            if sum(poseMAP(part_A,:))>0 && sum(poseMAP(part_B,:))>0
                %get mu, sigma and theta for generating the heatmaps
                [part_center, theta, len] = getHeatMapParams(poseMAP(part_A,:),poseMAP(part_B,:));
                
                %generate the heatmap
                pair_heatmap(:,:,j) = generateGHeatmap(pair_heatmap(:,:,j),part_center,theta,len,opts);
            end
            
            %generate weights for balancing positive / negative cells
            pair_heatmap_mask = getWeightMask(opts,insta,0,pair_heatmap,j,pair_heatmap_mask);
            
            %visualization
            %mapVisualize(opts, imt, sy, sx, pair_heatmap, j, poseMAP, tempY, poseRest);
            
        end
        
        %store labels
        labels{8,i} = pair_heatmap; %body part heatmap
        labels{9,i} = pair_heatmap_mask; %body part weihgting mask
    end
    
    labels{4,i} = insta;%number of instances (not used)
    %labels 5-7 and 12 are skipped (not used)
        
        %visualize the keypoints
        %drawkeypints(imt,sy,sx,tempY,tempRest);
        
        if ~isempty(opts.averageImage)
            offset = opts.averageImage ;
            if ~isempty(opts.rgbVariance)
                offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
            end
            imo(:,:,:,i) = bsxfun(@minus, imt(sy,sx,:), offset) ;
            imo(:,:,:,i) = imo(:,:,:,i)./256;
        else
            imo(:,:,:,i) = imt(sy,sx,:) ;
        end
        
        
    
    
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function transformations = generateTran(opts,batch)

%Generate a set of crop transformations for the data augmentation

tfs = [] ;
switch opts.transformation
    case 'none'
        tfs = [
            .5 ;
            .5 ;
            0 ] ;
    case 'flipOnly'
        tfs = [
            .5 .5;
            .5 .5;
            0  1] ;
    case 'f5'
        tfs = [...
            .5 0 0 1 1 .5 0 0 1 1 ;
            .5 0 1 0 1 .5 0 1 0 1 ;
            0 0 0 0 0  1 1 1 1 1] ;
    case 'f25'
        [tx,ty] = meshgrid(linspace(0,1,5)) ;
        tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
        tfs_ = tfs ;
        tfs_(3,:) = 1 ;
        tfs = [tfs,tfs_] ;
    case 'stretch'
    otherwise
        error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(batch)), 1) ;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [img, poseGT,poseRest, poseOccl, poseRestOccl] = parseData(opts,imdb,fr,i, im)
%load images and ground-truth
%poseGT: the main instance
%poseRest: rest instances

if isempty(im{i}) %nothing fetched
    %do nothing - to be added
    
else %data already in RAM
    img = single(im{i}) ;
    
    %keypoints ground-truth in cell format
    if size(imdb.images.labels{fr},3)>1 %multiple heatmap GT
        poseGT=imdb.images.labels{fr}(:,:,1);
        poseRest=imdb.images.labels{fr}(:,:,2:end);
        if opts.multipInst==0 %exclude multiple instances
            poseRest=[];
        end
    else
        poseGT=imdb.images.labels{fr};
        poseRest=[];
    end
    
end

%exclude occluded keypoints
if opts.inOcclud==0
    if opts.negHeat || opts.ignoreOcc
        poseOccl = double(poseGT(:,3) ==1) - double(poseGT(:,3) ==0); %visible 1, invisible -1 (not used anymore)
    else
        poseOccl = ones(size(poseGT,1));
    end
    
    poseGT(:,1)=poseGT(:,1).*poseGT(:,3);
    poseGT(:,2)=poseGT(:,2).*poseGT(:,3);
    poseGT=poseGT(:,1:2);
    
    if sum(size(poseRest))>0 %rest keyipoints
        if opts.negHeat || opts.ignoreOcc
            poseRestOccl=double(poseRest(:,3,:) ==1) - double(poseRest(:,3,:) ==0); %visible 1, invisible -1
        else
            poseRestOccl = ones(size(poseRest,1),size(poseRest,3));
        end
        poseRestOccl=squeeze(poseRestOccl);
        
        for k=1:size(poseRest,3)
            poseRest(:,1,k)=poseRest(:,1,k).*poseRest(:,3,k);
            poseRest(:,2,k)=poseRest(:,2,k).*poseRest(:,3,k);
        end
        poseRest=poseRest(:,1:2,:);
    end
    
else %include occluded keypoints
    if opts.negHeat || opts.ignoreOcc
        poseOccl = double(poseGT(:,3) ==1) - double(poseGT(:,3) ==0); %visible 1, invisible -1
    else
        poseOccl = ones(size(poseGT,1),1);
    end
    poseGT=poseGT(:,1:2);
    
    poseRestOccl=[];
    if sum(size(poseRest))>0 %rest keipoints
        if opts.negHeat || opts.ignoreOcc
            poseRestOccl=double(poseRest(:,3,:) ==1) - double(poseRest(:,3,:) ==0); %visible 1, invisible -1
        else
            poseRestOccl = ones(size(poseRest,1),size(poseRest,3));
        end
        poseRestOccl=squeeze(poseRestOccl);
        poseRest=poseRest(:,1:2,:);
    end
end

%ensure correct values for the main instance
idx=poseGT(:,1)>0; %zeros in x, means zeros in y as well
poseGT(idx,1) = max(1,poseGT(idx,1));
poseGT(idx,1) = min(size(img,2),poseGT(idx,1));
poseGT(idx,2) = max(1,poseGT(idx,2));
poseGT(idx,2) = min(size(img,1),poseGT(idx,2));
clear idx;

%ensure correct values for the rest instance
if sum(size(poseRest))>0
    for k=1:size(poseRest,3)
        idx=poseRest(:,1,k)>0; %zeros in x, means zeros in y as well
        poseRest(idx,1,k) = max(1,poseRest(idx,1,k));
        poseRest(idx,1,k) = min(size(img,2),poseRest(idx,1,k));
        poseRest(idx,2,k) = max(1,poseRest(idx,2,k));
        poseRest(idx,2,k) = min(size(img,1),poseRest(idx,2,k));
        clear idx;
    end
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function heatmap_mask = getWeightMask(opts,insta,thrs,heatmap,j,heatmap_mask)

if sum(sum(heatmap(:,:,j)))>0 %if eveything background, do nothing
    
    %include some background pixels in the heatmap mask
    %for balancing positive / negative cells
    if opts.HeatMapScheme==0
        heat_pixels=double(abs(heatmap(:,:,j))>0.0); %added abs for occlusion
        sideSize=sqrt(sum(heat_pixels(:)))/insta;
        SE = strel('square',round(sideSize*2));
        ex_heat_pixels = imdilate(heat_pixels,SE);
        integ_a=sum(heat_pixels(:));
        integ_b=numel(heatmap(:,:,j))-integ_a;
        heatmap_w=integ_a/integ_b;
        heatmap_mask(:,:,j) = ex_heat_pixels + heatmap_w;
    else
        %integral-idea
        heat_pixels=double(abs(heatmap(:,:,j))>thrs); %added abs for occlusion
        integ_a=sum(heat_pixels(:));
        integ_b=numel(heat_pixels(:,:))-integ_a;
        heatmap_wa= 1- (integ_a/numel(heatmap(:,:,j)));
        heatmap_wb= 1- (integ_b/numel(heatmap(:,:,j)));
        
        heat_pixels(heat_pixels<1) =heatmap_wb;%first this - order important
        heat_pixels(heat_pixels>=1)=heatmap_wa;
        
        %moda for the pairwise heatmaps
        %if more foreground pixels, do not weight
        if integ_a>integ_b
            heat_pixels=1;
        end
        
        heatmap_mask(:,:,j) = heat_pixels;
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [part_center, theta, len] = getHeatMapParams(keypA,keypB)

%center of the Gaussian
part_center(1,1) = keypA(1) + (keypB(1) - keypA(1))./2;
part_center(1,2) = keypA(2) + (keypB(2) - keypA(2))./2;

%rotation of the Gaussian
theta = atand ((keypB(2) - keypA(2)) /...
    (keypB(1) - keypA(1)));

%sigma of the gaussian based on the part length

len = norm(keypB(:) - keypA(:));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function mapVisualize(opts, imt, sy, sx, heatmap, j, poseMAP, tempY, poseRest)

disp('j is');
disp(j);

%visualization
tform = affine2d(opts.trf');
I=imwarp(uint8(imt(sy,sx,:)),tform);
padFact=[0,0];
if size(I,1)~=size(heatmap(:,:,j),1)
    padFact(1)=size(heatmap(:,:,j),1)-size(I,1);
end
if size(I,2)~=size(heatmap(:,:,j),2)
    padFact(2)=size(heatmap(:,:,j),2)-size(I,2);
end

%original image
imshow(uint8(imt(sy,sx,:))); hold on;
plot(tempY(j,1),tempY(j,2),'rx');hold off;

figure;
I = padarray(I,padFact,150,'pre');
I(:,:,2)=I(:,:,2)+uint8( double(rgb2gray(I)).*50.*(heatmap(:,:,j)-0.0) );
I(:,:,3)=I(:,:,3)+uint8( double(rgb2gray(I)).*10.*(heatmap(:,:,j)-0.0) );
imshow(I); hold on; plot(poseMAP(j,1),poseMAP(j,2),'rx');

if sum(size(poseRest))>0
    hold on;
    for k=1:size(poseRest,3)
        plot(poseRest(j,1,k),poseRest(j,2,k),'gx');
    end
    hold off;
end
figure; imagesc(heatmap(:,:,j));

pause(); close all;
%visualization

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function drawkeypints(imt,sy,sx,tempY,tempRest)

%plot the points
imshow(uint8(imt(sy,sx,:))); hold on;
for po=1:size(tempY,1)
    text(tempY(po,1),tempY(po,2), int2str(po),'Color','m','FontSize',15);
end
if sum(size(tempRest))>0
    for k=1:size(tempRest,3)
        tempY = tempRest(:,:,k);
        for po=1:size(tempY,1)
            text(tempY(po,1),tempY(po,2), int2str(po),'Color','g','FontSize',15);
        end
    end
end
hold off; pause();
%plot the points

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


