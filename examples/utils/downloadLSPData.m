function downloadLSPData()

%dowload MPII data and annotation(http://www.comp.leeds.ac.uk/mat4saj/lsp.html)
if ~exist('lsp_dataset', 'dir')
    url = 'http://www.comp.leeds.ac.uk/mat4saj/lsp_dataset.zip';
    fprintf('downloading %s\n', url) ;
    unzip(url) ;
end

%remap the ground-truth to the MPI format
mapp = [1,2,3,4,5,6,11,12,13,14,15,16,9,10];
Njo=14;

%full-body
par = [2,3,4,4,4,5,8,9,13,13,10,11,14,14]; %parent of each node
coordMap=[13,13,13,13,13,13,9,10,3,4,8,11,2,5;...
    13,14, 9, 3, 4,10,8,11,2,5,7,12,1,6];

flipFlg='full';


%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%
%Dataset
rootPth='lsp_dataset/';
datas='LSP';
load([rootPth 'joints.mat']);
joints=permute(joints,[2 1 3]);
jointsTr=joints(:,:,1:1000);
jointsVal = joints(:,:,1001:end);
imgFmt=[rootPth 'images/im%.4d.jpg'];
clear joints;
%%%%%%%%%%%%%%%% LSP %%%%%%%%%%%%%%%%

imgSize = [256,256];

cnt=0;
h = waitbar(0,'Please wait...');

disp('Please Wait...');
for i=1:size(jointsTr,3)
    cnt = cnt + 1;
    
    img = imread(sprintf(imgFmt,i));
    
    %             %visualization
    %             imshow(img); hold on;
    %             pose=jointsTr{d}(:,:,i);
    %             for jj=1:1:size(pose,1) %active indiv.
    %                 if pose(jj,3)==1
    %                     text(pose(jj,1),pose(jj,2),int2str(jj),'Color','m','FontSize',16);
    %                 else
    %                     text(pose(jj,1),pose(jj,2),int2str(jj),'Color','r','FontSize',16);
    %                 end
    %             end
    %             pause();
    %             hold off;
    %             close all;
    
    %poseGT=jointsTr(:,:,i); %2D GT
    poseGT = zeros(16,3);
    for j=1:numel(mapp)
        poseGT(mapp(j),1:3) = jointsTr(j,:,i); %2D GT
    end
    
    %missing - occluded GT
    idx=(poseGT(:,1)<=1 & poseGT(:,2)<=1 & poseGT(:,3)==0);
    poseGT(idx,:) = 0;
    
    %all image
    xUpLe = 0;
    yUpLe = 0;
    wid = size(img,2);
    hei = size(img,1);
    
    %check if the bounding box exceeds the image plane
    %and pad the image and all GT poses
    padUpX = max(ceil((hei-wid)/2),0);
    padLoX = max(ceil((hei-wid)/2),0);
    padUpY = max(ceil((wid-hei)/2),0);
    padLoY = max(ceil((wid-hei)/2),0);
    
    wid = wid + padUpX + padLoX;
    hei = hei + padUpY + padLoY;
    
    imgPad = uint8(128*ones(padUpY+size(img,1)+padLoY, ...
        padUpX+size(img,2)+padLoX,3));
    imgPad = uint8(imgPad);
    imgPad(1+padUpY:padUpY+size(img,1),1+padUpX:padUpX+size(img,2),:) = ...
        img;
    
    bbox{cnt} = [xUpLe, yUpLe, wid, hei];
    pad_train{cnt} = [padUpX, padUpY, padLoX, padLoY];
    ptsAll{cnt}=poseGT;
    
    %crop the image
    imgPath{cnt}=imgPad;
    
    %change the origin for the padded image
    idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
    ptsAll{cnt}(idx,1)=ptsAll{cnt}(idx,1)+padUpX;
    ptsAll{cnt}(idx,2)=ptsAll{cnt}(idx,2)+padUpY;
    
    %shift the origin for the active individual
    ptsAll{cnt}(:,1)=ptsAll{cnt}(:,1)-xUpLe;
    ptsAll{cnt}(:,2)=ptsAll{cnt}(:,2)-yUpLe;
    checkX=double(ptsAll{cnt}(:,1)>0);
    checkY=double(ptsAll{cnt}(:,2)>0);
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
    checkX=double(ptsAll{cnt}(:,1)<size(imgPath{cnt},2));
    checkY=double(ptsAll{cnt}(:,2)<size(imgPath{cnt},1));
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
    
    %resize to standard size
    s_s = [size(imgPath{cnt},1) size(imgPath{cnt},2)];
    s_t = [imgSize(1) imgSize(2)];
    s = s_s.\s_t;
    tf = [ s(2) 0 0; 0 s(1) 0; 0  0 1];
    T = affine2d(tf);
    
    %points scaled
    [ptsAll{cnt}(:,1),ptsAll{cnt}(:,2)] = transformPointsForward(T, ptsAll{cnt}(:,1),ptsAll{cnt}(:,2));
    
    %image resized
    imgPath{cnt} = imresize(imgPath{cnt}, 'scale', s, 'method', 'bilinear');
    
    %     %visualization
    %     imshow(imgPath{cnt}); hold on;
    %     x=size(imgPath{cnt},2)/2;
    %     y=size(imgPath{cnt},1)/2;
    %     text(x,y,'C','Color','m','FontSize',22);
    %     poseGT=ptsAll{cnt};
    %     for jj=1:1:size(poseGT,1) %active indiv.
    %         if poseGT(jj,3)==1
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
    %         else
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
    %         end
    %     end
    %     pause();
    %     hold off; close;
    %
    %
    %     img = imread(sprintf(imgFmt,i));%1001 first image
    %     poseGT=ptsAll{cnt};
    %
    %     [poseGT(:,1),poseGT(:,2)] = transformPointsInverse(T, poseGT(:,1),poseGT(:,2));
    %
    %     idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
    %     poseGT(idx,1) = poseGT(idx,1) + xUpLe;
    %     poseGT(idx,2) = poseGT(idx,2) + yUpLe;
    %     poseGT(idx,1) = poseGT(idx,1) - padUpX;
    %     poseGT(idx,2) = poseGT(idx,2) - padUpY;
    %
    %     imshow(img); hold on;
    %     for jj=1:1:size(poseGT,1) %active indiv.
    %         if poseGT(jj,3)==1
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
    %         else
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
    %         end
    %     end
    %     pause();
    %     hold off;
    
    waitbar(i / size(jointsTr,3));
end

save('lsp_dataset/Train.mat','imgPath','ptsAll','bbox','pad_train','-v7.3'); %pose data

clear imgPath ptsAll bbox

%%Validation
cnt = 0 ;
for i=1:size(jointsVal,3)
    cnt = cnt + 1;
    
    img = imread(sprintf(imgFmt,i+1000));%1001 first image
    
    %poseGT=jointsTr(:,:,i); %2D GT
    poseGT = zeros(16,3);
    for j=1:numel(mapp)
        poseGT(mapp(j),1:3) = jointsVal(j,:,i); %2D GT
        %1 stands for occluded, make 1 for visible
        poseGT(mapp(j),3) = 1 - poseGT(mapp(j),3);
    end
    
    %missing - occluded GT
    idx=(poseGT(:,1)<=1 & poseGT(:,2)<=1 & poseGT(:,3)==0);
    poseGT(idx,:) = 0;
    
    %fit a bounding box
    idx = (poseGT(:,1)>0 & poseGT(:,2)>0);
    xUpLe=min(poseGT(idx,1));
    yUpLe=min(poseGT(idx,2));
    wid=max(poseGT(idx,1))-min(poseGT(idx,1));
    hei=max(poseGT(idx,2))-min(poseGT(idx,2));
    
    %all image
    xUpLe = 0;
    yUpLe = 0;
    wid = size(img,2);
    hei = size(img,1);
    
    %check if the bounding box exceeds the image plane
    %and pad the image and all GT poses
    padUpX = max(ceil((hei-wid)/2),0);
    padLoX = max(ceil((hei-wid)/2),0);
    padUpY = max(ceil((wid-hei)/2),0);
    padLoY = max(ceil((wid-hei)/2),0);
    
    wid = wid + padUpX + padLoX;
    hei = hei + padUpY + padLoY;
    
    imgPad = uint8(128*ones(padUpY+size(img,1)+padLoY, ...
        padUpX+size(img,2)+padLoX,3));
    imgPad = uint8(imgPad);
    imgPad(1+padUpY:padUpY+size(img,1),1+padUpX:padUpX+size(img,2),:) = ...
        img;
    
    bbox{cnt} = [xUpLe, yUpLe, wid, hei];
    pad_val{cnt} = [padUpX, padUpY, padLoX, padLoY];
    ptsAll{cnt}=poseGT;
    
    %crop the image
    imgPath{cnt}=imgPad;
    
    %change the origin for the padded image
    idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
    ptsAll{cnt}(idx,1)=ptsAll{cnt}(idx,1)+padUpX;
    ptsAll{cnt}(idx,2)=ptsAll{cnt}(idx,2)+padUpY;
    
    %shift the origin for the active individual
    ptsAll{cnt}(:,1)=ptsAll{cnt}(:,1)-xUpLe;
    ptsAll{cnt}(:,2)=ptsAll{cnt}(:,2)-yUpLe;
    checkX=double(ptsAll{cnt}(:,1)>0);
    checkY=double(ptsAll{cnt}(:,2)>0);
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
    checkX=double(ptsAll{cnt}(:,1)<size(imgPath{cnt},2));
    checkY=double(ptsAll{cnt}(:,2)<size(imgPath{cnt},1));
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkX checkX];
    ptsAll{cnt}(:,1:2)= ptsAll{cnt}(:,1:2).*[checkY checkY];
    
    %resize to standard size
    s_s = [size(imgPath{cnt},1) size(imgPath{cnt},2)];
    s_t = [imgSize(1) imgSize(2)];
    s = s_s.\s_t;
    tf = [ s(2) 0 0; 0 s(1) 0; 0  0 1];
    T = affine2d(tf);
    
    %points scaled
    [ptsAll{cnt}(:,1),ptsAll{cnt}(:,2)] = transformPointsForward(T, ptsAll{cnt}(:,1),ptsAll{cnt}(:,2));
    
    %image resized
    imgPath{cnt} = imresize(imgPath{cnt}, 'scale', s, 'method', 'bilinear');
    
    %     %visualization
    %     imshow(imgPath{cnt}); hold on;
    %     x=size(imgPath{cnt},2)/2;
    %     y=size(imgPath{cnt},1)/2;
    %     text(x,y,'C','Color','m','FontSize',22);
    %     poseGT=ptsAll{cnt};
    %     for jj=1:1:size(poseGT,1) %active indiv.
    %         if poseGT(jj,3)==1
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
    %         else
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
    %         end
    %     end
    %     pause();
    %     hold off; close;
    %
    %
    %     img = imread(sprintf(imgFmt,i+1000));%1001 first image
    %     poseGT=ptsAll{cnt};
    %
    %     [poseGT(:,1),poseGT(:,2)] = transformPointsInverse(T, poseGT(:,1),poseGT(:,2));
    %
    %     idx=(ptsAll{cnt}(:,1)>0 & ptsAll{cnt}(:,2)>0);
    %     poseGT(idx,1) = poseGT(idx,1) + xUpLe;
    %     poseGT(idx,2) = poseGT(idx,2) + yUpLe;
    %     poseGT(idx,1) = poseGT(idx,1) - padUpX;
    %     poseGT(idx,2) = poseGT(idx,2) - padUpY;
    %
    %     imshow(img); hold on;
    %     for jj=1:1:size(poseGT,1) %active indiv.
    %         if poseGT(jj,3)==1
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','m','FontSize',16);
    %         else
    %             text(poseGT(jj,1),poseGT(jj,2),int2str(jj),'Color','r','FontSize',16);
    %         end
    %     end
    %     pause();
    %     hold off;
    
end
save('lsp_dataset/Validation.mat','imgPath','bbox','ptsAll','pad_val','-v7.3'); %pose data
disp('Completed...');
delete(h);

end