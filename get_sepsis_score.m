function [scores, labels]  = get_sepsis_score(rawdata,model)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



data=rawdata(:,[1:36 39:40]);

xxx=find(data(:,4)<data(:,6));

for ii=1:length(xxx)
    uu=data(xxx(ii),4);
    vv=data(xxx(ii),6);
    data(xxx(ii),6)=uu;
    data(xxx(ii),4)=vv;
end

[data] = fillthenans1(data);
data(isnan(data))=0;
rdata=data;
feat=propfeat(data);
data1=[data feat];
% x=[5,4,4,6,-2,2,-1,4,8,0,8,4,0,1,2,7,6,0,-2,-3,-1,0,2,5,4,2,5,3,3,-1,3,7,4,2,7,1,-2,-1,4,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1];
% power = x(1:39);
% data2=data1.^power;
x=[5,4,4,6,-2,2,-1,4,8,0,8,4,0,1,2,7,6,0,-2,-3,-1,0,2,5,4,2,5,3,3,-1,3,7,4,2,7,1,-2,-1,4,4,3,2,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,0];
power = x(1:42);
data2=data1.^power;
data22=data2(:,logical(x(43:end)));
data33=[data1 data22(:,[1:6 8:9 11:end])];
y2=[1,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1];
data3=data33(:,logical(y2));
stats = statisticalfeatures(rdata(:,[1:8]));

%shock_index=data(:,1)./(data(:,4).*data(:,35));
CC=[1,2,9,2,8,10,1,10,3,3,29,14,12,4,14,17,10,8,15,13,6,7,14,1,12,7;37,12,24,25,36,16,9,23,32,35,32,33,37,27,27,34,37,32,37,19,33,19,36,11,32,23];
data111=data(:,CC(1,:)).^3./(1-data(:,CC(2,:)));
CC11=[13,14,14,8,14,1,6,13,23,17,8,10,4,3,6,9,5,5,5,10,5,17,9,2,16,1,8,3,1,21,2,3,19,4,10,11,10,5;19,33,31,23,34,15,23,36,37,21,34,30,33,29,33,20,33,37,37,26,24,36,30,7,34,35,27,27,30,33,24,11,28,37,36,21,15,18];
data41=data(:,CC11(1,:)).^3./(1-data(:,CC11(2,:)).^2);
CC22=[20,10,10,18,2,18,1,10,6,17,6,17,19,8,5,12,12,7,19,12,3,1;34,37,32,33,25,30,22,28,17,27,35,25,37,36,32,35,37,31,34,32,33,28];
data48=data(:,CC22(1,:))./(1-data(:,CC22(2,:)).^3);
data4=[  data111 data41 data48];
y3=[1,0,0,1,0,0,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,1,0,0,0,0,0,0,0];
y4=[0,1,1,0,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,0,1,0,1,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,1,0,1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,0,1,0,1,0,1];
finaldata=[data3(end,:) data4(end,logical(y3)) stats(:,logical(y4))];


[labels1,scores1] = predict(model{1},finaldata(end,:));
 [labels2,scores2] = predict(model{2},finaldata(end,:));
 [labels3,scores3] = predict(model{3},finaldata(end,:));
 [labels4,scores4] = predict(model{4},finaldata(end,:));
 [labels5,scores5] = predict(model{5},finaldata(end,:));

 scoresf=[scores1(end,2) scores2(end,2) scores3(end,2) scores4(end,2) scores5(end,2)];

labelsf=[labels1(end) labels2(end) labels3(end) labels4(end) labels5(end)];

scores=mode(scoresf,2);
labels=mode(labelsf,2);

 
 
end




function [outputdata] = fillthenans1(inputdata)
 outputdataf=[];
 if(size(inputdata,1)==1)
     outputdataf=inputdata;
 else
        
        for uu=1:38
        datata=fillmissing(inputdata(:,uu)','linear',2,'EndValues','nearest');
        outputdataf(:,uu)=datata';
        end
        
 end
outputdata=outputdataf;

end

function feat=propfeat(data)

feat11=data(:,1)./(data(:,35).^2);%correct
feat12=data(:,6)./(data(:,36).^2);
feat13=data(:,8)./(data(:,31).^2);
feat14=data(:,35)./(data(:,36).^2);




feat=[feat11 feat12 feat13 feat14  ];
feat(isinf(feat))=100;



end
function [features] = statisticalfeatures(inputdata)

% inputdata is 1:34 elements after nanfill

diffvar=[];diffvar1=[];othervar1=[];othervar2=[];othervar=[];othervar11=[];

 J=size(inputdata,1);
 try
 for ii=1:8
    diffvar=[wentropy(inputdata(:,ii),'shannon') sum(inputdata(:,ii).^2)./J mean(diff(inputdata(:,ii)))];
    diffvar1=[diffvar1 diffvar];
 end
 catch
     diffvar1=zeros(24,1)';
 end
 
 
 try
 for ii=1:8
     if(J>5)
    qdata=quantile(inputdata(end-5+1:end,ii),[0.01 0.05 .95 0.99]);
    othervar=[mean(qdata) max(qdata) min(qdata) median(qdata) var(qdata)];
    
     else
         othervar=zeros(5,1)';
     end
     othervar1=[othervar1 othervar];
 end
 catch
     othervar1=zeros(40,1)';
 end
 
 try
 for ii=1:8
     if(J>11)
    qdata=quantile(inputdata(end-11+1:end,ii),[0.01 0.05 .95 0.99]);
    othervar11=[mean(qdata) max(qdata) min(qdata) median(qdata) var(qdata)];
    
     else
         othervar11=zeros(5,1)';
     end
     othervar2=[othervar2 othervar11];
 end
 catch
     othervar2=zeros(40,1)';
 end
 
 features=[diffvar1 othervar1 othervar2];
    

end