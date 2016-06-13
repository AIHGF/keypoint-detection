function y= flipKeyPointsCoords(leng,poseGTresc,jnts)

if strcmp(jnts,'joint')==1
    y=poseGTresc;
    y(1) = - y(1);
elseif strcmp(jnts,'head')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(2,:);
    poseGTresc(2,:)=temp(1,:);
    y=poseGTresc;
    
elseif strcmp(jnts,'torso')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(6,:);
    poseGTresc(6,:)=temp(1,:);
    poseGTresc(2,:)=temp(5,:);
    poseGTresc(5,:)=temp(2,:);
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    y=poseGTresc;
    
elseif strcmp(jnts,'hands')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(4,:);
    poseGTresc(4,:)=temp(1,:);
    poseGTresc(2,:)=temp(3,:);
    poseGTresc(3,:)=temp(2,:);
    y=poseGTresc;
    
elseif strcmp(jnts,'torsoOnly')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(2,:);
    poseGTresc(2,:)=temp(1,:);
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    y=poseGTresc;
elseif strcmp(jnts,'legs')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(6,:);
    poseGTresc(6,:)=temp(1,:);
    poseGTresc(2,:)=temp(5,:);
    poseGTresc(5,:)=temp(2,:);
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    y=poseGTresc;
    
elseif strcmp(jnts,'upper')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    poseGTresc(2,:)=temp(5,:);
    poseGTresc(5,:)=temp(2,:);
    poseGTresc(1,:)=temp(6,:);
    poseGTresc(6,:)=temp(1,:);
    %poseGTresc = reshape (poseGTresc',size(poseGTresc,1)*2,1);
    y=poseGTresc;
    
elseif strcmp(jnts,'bbc')==1
    idx = poseGTresc(:,1)>0;
    poseGTresc(idx,1)=leng-poseGTresc(idx,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(2,:)=temp(3,:);
    poseGTresc(3,:)=temp(2,:);
    poseGTresc(4,:)=temp(5,:);
    poseGTresc(5,:)=temp(4,:);
    poseGTresc(6,:)=temp(7,:);
    poseGTresc(7,:)=temp(6,:);
    %poseGTresc = reshape (poseGTresc',size(poseGTresc,1)*2,1);
    y=poseGTresc;
    
elseif strcmp(jnts,'mpi')==1
    idx = poseGTresc(:,1)>0;
    poseGTresc(idx,1)=leng-poseGTresc(idx,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(6,:);
    poseGTresc(6,:)=temp(1,:);
    poseGTresc(2,:)=temp(5,:);
    poseGTresc(5,:)=temp(2,:);
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    poseGTresc(13,:)=temp(14,:);
    poseGTresc(14,:)=temp(13,:);
    poseGTresc(12,:)=temp(15,:);
    poseGTresc(15,:)=temp(12,:);
    poseGTresc(11,:)=temp(16,:);
    poseGTresc(16,:)=temp(11,:);
    y=poseGTresc;

elseif strcmp(jnts,'mpiUp')==1
    idx = poseGTresc(:,1)>0;
    poseGTresc(idx,1)=leng-poseGTresc(idx,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(4,:)=temp(9,:);
    poseGTresc(9,:)=temp(4,:);
    poseGTresc(5,:)=temp(8,:);
    poseGTresc(8,:)=temp(5,:);
    poseGTresc(6,:)=temp(7,:);
    poseGTresc(7,:)=temp(6,:);
    y=poseGTresc;
    
elseif strcmp(jnts,'box')==1
    poseGTresc(:,1)=leng-poseGTresc(:,1); %change origin
    y=poseGTresc;
    wid=y(1,1)-y(2,1);    
    y(1,1)=y(1,1) - wid;
    y(2,1)=y(2,1) + wid;
    
else %full body (LSP)
    idx = poseGTresc(:,1)>0;
    poseGTresc(idx,1)=leng-poseGTresc(idx,1); %change origin
    temp=poseGTresc;
    poseGTresc=temp;
    poseGTresc(1,:)=temp(6,:);
    poseGTresc(6,:)=temp(1,:);
    poseGTresc(2,:)=temp(5,:);
    poseGTresc(5,:)=temp(2,:);
    poseGTresc(3,:)=temp(4,:);
    poseGTresc(4,:)=temp(3,:);
    poseGTresc(7,:)=temp(12,:);
    poseGTresc(12,:)=temp(7,:);
    poseGTresc(8,:)=temp(11,:);
    poseGTresc(11,:)=temp(8,:);
    poseGTresc(9,:)=temp(10,:);
    poseGTresc(10,:)=temp(9,:);
    y=poseGTresc;
end

end