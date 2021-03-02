

%% 4

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


%% K fold
% randmlize the data sets
seed = randperm(length(YTrain));
XTrain = XTrain(seed,:);
YTrain = YTrain(seed);


% split into k group
k = 5;
s = length(YTrain)/k;
for i=1:k
    X{i} = XTrain((i-1)*s+1:i*s,1:end);
    Y{i} = YTrain((i-1)*s+1:i*s);
end
clearvars -except XTrain YTrain  XTest YTest k X Y

% SVM
C = [1e-6, 1e-5, 1e-3, 1e-2, 1e-1,0.5, 1, 1e1, 1e2, 1e3]; % find optimal C using k-fold
gama = [1e-6, 1e-5, 1e-3, 1e-2, 1e-1, 0.5 , 1, 1e1, 1e2, 1e3];

% init
mean_accuracies_each_gama = zeros(length(gama),1);
accuracies_in_test = zeros(length(gama),1); % for each gamma
optimal_C = zeros(length(gama),1);
% main
for i = 1:length(gama)
    % select a gama 
    mean_acc = zeros(length(C),1);
    for j = 1:length(C) % find optimal C using K fold
        % and perform k-fold
        accuracies_in_k = zeros(length(k),1);
        parfor n = 1:k
        % k fold
            XTrain_temp = [];
            YTrain_temp = [];
            XTest_temp = X{n}; 
            YTest_temp = Y{n};
            % assemble X and Y
            for m = 1:k
                if m == n
                    continue
                end
                XTrain_temp = [XTrain_temp; X{m}];
                YTrain_temp = [YTrain_temp; Y{m}]; 
            end% end of data sets reconstruction
            % train
            model = svmtrain(YTrain_temp,XTrain_temp,...
                [' -s 0 -t 1 -c ', convertStringsToChars(string(C(j))),...
                 ' -g ', convertStringsToChars(string(gama(i)))]);
             [predicted_label, accuracy, dec_values_L] = svmpredict(YTest_temp, XTest_temp,model);
            accuracies_in_k(n) = accuracy(1);
        end % end of k fold
        mean_acc(j) = mean(accuracies_in_k);
    end
    [max_C,C_idx] = max(mean_acc);
    
    model = svmtrain(YTrain_temp,XTrain_temp,...
                [' -s 0 -t 1 -c ', convertStringsToChars(string(C(C_idx))),...
                 ' -g ', convertStringsToChars(string(gama(i)))]);
    mean_accuracies_each_gama(i) = max_C;cm
    optimal_C(i) = C(C_idx);
    [predicted_label, accuracy, dec_values_L] = svmpredict(YTest, XTest,model);
    accuracies_in_test(i) = accuracy(1);
    models{i} = model;
end
save('4_SVM_RBK_K_fold.mat')



%% 3 
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


%% K fold
% randmlize the data sets
seed = randperm(length(YTrain));
XTrain = XTrain(seed,:);
YTrain = YTrain(seed);


% split into k group
k = 5;
s = length(YTrain)/k;
for i=1:k
    X{i} = XTrain((i-1)*s+1:i*s,1:end);
    Y{i} = YTrain((i-1)*s+1:i*s);
end
clearvars -except XTrain YTrain  XTest YTest k X Y


%% SVM C-SVM
C = [1e-6, 1e-5, 1e-3, 1e-2, 1e-1,0.5, 1, 1e1, 1e2, 1e3];

SVM_accuracies_in_test = zeros([1,length(C)]);
SVM_accuracies_in_train = zeros([1,length(C)]);
SVM_accuracies_in_train_mean = zeros([1,length(C)]);
SVMaccuracies = zeros([1,k]);


for n = 1:length(C)
    for i = 1:k
        XTrain_temp = [];
        YTrain_temp = [];
        XTest_temp = X{i};
        YTest_temp = Y{i};
        % assemble X and Y
        for j = 1:k
            if j == i
                continue
            end
            XTrain_temp = [XTrain_temp; X{j}];
            YTrain_temp = [YTrain_temp; Y{j}]; 
        end
        % train
        mdl = svmtrain(YTrain_temp,XTrain_temp,[' -s 0 -t 1 -c ', convertStringsToChars(string(C(n)))]);
        [predicted_label, accuracy, dec_values_L] = svmpredict(YTest_temp, XTest_temp,mdl);
        SVMaccuracies(i) = accuracy(1);
    end
    SVM_accuracies_in_train_mean(n) = sum(SVMaccuracies)/k;
    % re-train the modle using all samples
    mdl = svmtrain(YTrain,XTrain,[' -s 0 -t 1 -c ', convertStringsToChars(string(C(n)))]);
    SVMmdls{n} = mdl;
    [predicted_label, accuracy, dec_values_L] = svmpredict(YTest, XTest,mdl);
    SVM_accuracies_in_test(n) = accuracy(1);
    [predicted_label, accuracy, dec_values_L] = svmpredict(YTrain, XTrain,mdl);
    SVM_accuracies_in_train(n) = accuracy(1);
    
    
end
save("p3_k_fold_SVM.mat")

%% Logistic regression
clearvars -except XTrain YTrain  XTest YTest k X Y
lambda = [1e-16,1e-15,1e-14,1e-13,1e-12,1e-11,1e-10, 1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2,0.05, 1e-1, 1, 1e1, 1e2, 1e3];
for i = 1:k
    XTrain_temp = [];
    YTrain_temp = [];
    XTest_temp = X{i}; 
    YTest_temp = Y{i};
    % assemble X and Y
    for j = 1:k
        if j == i
            continue
        end
        XTrain_temp = [XTrain_temp; X{j}];
        YTrain_temp = [YTrain_temp; Y{j}]; 
    end
    % train
    mdl = fitclinear(XTrain_temp, YTrain_temp,"Learner","logistic","Regularization","lasso", "Lambda",lambda,"Solver","sgd");
    Label = predict(mdl,XTest_temp);
    LRaccuracies(i,:) = sum(YTest_temp==Label)./length(YTest_temp);
end
LR_accuracies_in_train_mean = sum(LRaccuracies)/k;
% re-train the modle using all samples
mdl = fitclinear(XTrain, YTrain,"Learner","logistic","Regularization","lasso", "Lambda",lambda,"Solver","sgd");
Label = predict(mdl,XTest);
LRaccuracies = sum(YTest==Label)/length(Label);
save("p3_k_fold_LR.mat")

