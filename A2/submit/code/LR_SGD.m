% load 
filenameImagesTrain = 'train-images-idx3-ubyte';
filenameLabelsTrain = 'train-labels-idx1-ubyte';
filenameImagesTest = 't10k-images-idx3-ubyte';
filenameLabelsTest = 't10k-labels-idx1-ubyte';

XTrain_all = processImagesMNIST(filenameImagesTrain);
YTrain_all = processLabelsMNIST(filenameLabelsTrain);
XTest_all= processImagesMNIST(filenameImagesTest);
YTest_all = processLabelsMNIST(filenameLabelsTest);
% Select label 5 and 7
XTrain = XTrain_all(YTrain_all==5|YTrain_all==7,:);
YTrain = YTrain_all(YTrain_all==5|YTrain_all==7,:);
XTest = XTest_all(YTest_all==5|YTest_all==7,:);
YTest = YTest_all(YTest_all==5|YTest_all==7,:);
YTest(YTest==5)=1;
YTrain(YTrain==5)=1;
YTest(YTest==7)=0;
YTrain(YTrain==7)=0;

% 
clearvars -except XTrain YTrain XTest YTest
global t alpha t0 d r C learning_rate_init
close all


%%  main
% Cost function parameter
C = 20;
% number of epoch
no_epoch = 5;
% %% learn rate update
% For learn rate: 
% 1:Time-based  
% 2:Step-based; 
% any other number: constant learn rate. eg: 1e-3
lr_setup = 2;
t = 1;
t0 = 0.1;
alpha = 0.1;

% another schulder
d = 0.7; % learning rate should change at each drop
r = 500; % often the rate should be dropped
learning_rate_init = 0.5;

%%%%%%%%%%%nothing need to be change from here%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = [];
learn_rate = [];
w=rand(size(XTrain(1,:)));
b= rand(1);
for i =1:no_epoch 

    
    [w,b,l,lr] = LR_SGD_fit(w, b, XTrain, YTrain, lr_setup); 
    loss = [loss,l];
    learn_rate = [learn_rate, lr];
end

[y_hat,prob,accuracy]=LR_SGD_predict(w,b,XTest,YTest);
accuracy
figure
plot(loss)
title('loss')
figure
plot(learn_rate)
title('learn rate')
% 



