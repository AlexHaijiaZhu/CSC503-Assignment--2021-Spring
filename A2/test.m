clc
close all
clear all
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
tt = XTest(1,:)';
tt=reshape(tt.*255,[28,28]);
imshow(tt');
%%
X=XTrain;
Y=YTrain;
w = rand([1,length(X(1,:))])
b = rand(1)
for i =1:10 
[w,b,loss,lr] = LR_re_opt(w, b, X, Y,1, length(YTrain));
end

y_hat=pd(w,b,XTest);
sum(y_hat==YTest)/length(YTest);
plot(loss)
figure
plot(lr)
%% --------------------------------------------------
% Predict

function [label] = pd(w,b,X)
label=ones([1,length(X)]);
for i = 1: length(label)
    label(i) = my_sigmod(dot(w,X(i,:))+b);    
end

label(label>0.5)=1;
label(label<0.5)=0;
if sum(label(label==0.5))>0
    disp("find some label in the middle")
end
end


%% Model optimization
%%
function [w,b,loss,lr] = LR_re_opt(w, b, X, Y,init_learning_rate, no_iterations)
loss = zeros([1,no_iterations]);
lr = zeros([1,no_iterations]);
    for i =1:no_iterations
        % predicted result
        Y_hat = my_sigmod(dot(w,X(i,:))+b);
        cost = loss_cross_entropy(Y_hat,Y(i,:));% + 1/2*norm(w);

        % Gradient calculation
        dw = (Y(i,:)-Y_hat).*X(i,:);
        db = (Y(i,:)-Y_hat).*X(i,:);
        
        d =0.77;% learning rate should change at each drop
        r=50; %often the rate should be dropped
        learning_rate = init_learning_rate * d^floor((1+i/r));
        %% update parameter 
        w = w + learning_rate.*dw./length(X(i,:));
        b = b + sum(learning_rate.*db)./length(X(i,:));
        
        loss(i)=cost;
        lr(i) = learning_rate;
    end    
end


%% ----------------------------Working!-----------------------------------
function loss = loss_cross_entropy(y_estimate, y_actual) 
    loss = -y_actual*log(y_estimate)-(1-y_actual)*log(1-y_estimate);
end

function prob = my_sigmod(z)    
    prob = 1/(1+exp(-z));
end