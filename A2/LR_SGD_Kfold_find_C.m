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
global t alpha t0 d r C learning_rate_init
Cs =[1e-3,1e-2,1e-1, 1, 5, 1e1,5e1,1e2,5e2,1e3,5e3,1e4,1e5,1e6,1e7];
t0s = [1e-3, 1e-2, 1e-1,1, 5,1e1,1e2,5e2,1e4, 5e4];
alphas = [1e-1,1, 5,1e1,1e2,5e2,1e4, 5e4];
no_epoch = 5;
% init
mean_accuracies = zeros(length(Cs),1);
test_accuracy = zeros(length(Cs),1);
accuracies_in_test = zeros(length(Cs),1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% main
% Cost function parameter

% number of epoch

% %% learn rate update
% For learn rate: 
% 1:Time-based  
% 2:Step-based; 
% any other number: constant learn rate. eg: 1e-3
lr_setup = 1;
t = 1;
t0 = 5e4;
alpha = 5;

figure(1)
hold on 
grid on
for j = 1:length(Cs) % find optimal t0 using K fold
    C = Cs(j)
    % and perform k-fold

    accuracies_in_k = zeros(length(k),1);
    for n = 1:k
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
        loss = [];
        learn_rate = [];
        w=rand(size(XTrain(1,:)));
        b= rand(1);
        t = 1;
        for num_e =1:no_epoch 
            [w,b,l,lr] = LR_SGD_fit(w, b, XTrain_temp, YTrain_temp, lr_setup); 
            loss = [loss,l];
            learn_rate = [learn_rate, lr];
        end
%             loss_in_cell{j,n} = loss;
        [y_hat,prob,accuracy]=LR_SGD_predict(w,b,XTest_temp,YTest_temp);
        accuracies_in_k(n)= accuracy;
    end % end of k fold
    mean_accuracies(j) = mean(accuracies_in_k);
    % retrain 
    loss = [];
    learn_rate = [];
    w=rand(size(XTrain(1,:)));
    b= rand(1);
    t = 1;
    for num_e =1:no_epoch 
        [w,b,l,lr] = LR_SGD_fit(w, b, XTrain, YTrain, lr_setup); 
        loss = [loss,l];
        learn_rate = [learn_rate, lr];
    end
    [y_hat,prob,accuracy]=LR_SGD_predict(w,b,XTest,YTest);
    test_accuracy(j)=accuracy;
%     [max_C,C_idx] = max(mean_acc);
end %alpha

%%
hold off
plot(test_accuracy,'linewidth',2)
hold on 
plot(mean_accuracies,'linewidth',2)
grid on
legend('Validation Accuracy','Train Accuracy')
xlabel('index','fontsize',16)
ylabel('Accuracy (%)','fontsize',16)
ylim([50,100])
