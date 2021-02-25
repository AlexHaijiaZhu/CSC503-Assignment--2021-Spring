%% load 
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
test = XTest(1,:)';
test=reshape(test.*255,[28,28]);
imshow(test')

%% K fold
% randmlize the data sets
seed = randperm(length(YTrain));
XTrain_rand = XTrain(seed,:);
YTrain_rand = YTrain(seed);
clearvars -except YTrain_rand XTrain_rand

% split into k group
k = 10;
s = length(YTrain_rand)/k;
for i=1:k
    X{i} = XTrain_rand((i-1)*s+1:i*s,1:end);
    Y{i} = YTrain_rand((i-1)*s+1:i*s);
end

% perfrom CV on the "new taining/test set", and repeat

%% logistic regression
accuracies = zeros([1,k]);
for i = 1:k
    XTrain = [];
    YTrain = [];
    XTest = X{i};
    YTest = Y{i};
    for j = 1:k
        if j == i
            continue
        end
        XTrain = [XTrain; X{j}];
        YTrain = [YTrain; Y{j}]; 
    end
    mdl = fitclinear(XTrain, YTrain,"Learner","logistic","Regularization","lasso", "Lambda",1e-5,"Solver","sgd");
    Label = predict(mdl,XTest);
    accuracy = sum(Label == YTest)/length(YTest);
    LRaccuracies(i) = accuracy;
    LRmdls{i} = mdl;
end

mean_accuracy = sum(LRaccuracies)/k


%% SVM C-SVM
accuracies = zeros([1,k]);
for i = 1:k
    XTrain = [];
    YTrain = [];
    XTest = X{i};
    YTest = Y{i};
    for j = 1:k
        if j == i
            continue
        end
        XTrain = [XTrain; X{j}];
        YTrain = [YTrain; Y{j}]; 
    end
    mdl = svmtrain(YTrain,XTrain,['-s 0 -t 0 -c 2']);
    [predicted_label, accuracy, dec_values_L] = svmpredict(YTest, XTest,mdl);
    SVMaccuracies(i) = accuracy(1);
    SVMmdls{i} = mdl;
end

mean_accuracy = sum(SVMaccuracies)/k