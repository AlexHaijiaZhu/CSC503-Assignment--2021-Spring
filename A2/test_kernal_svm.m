% Fourth: Gaussian kernel
gama = [1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, eps, 1, 1e1, 1e2, 1e3];
C = [1e-9,1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, eps, 1, 1e1, 1e2, 1e3];
accuracies = zeros(length(gama),length(C));
traing_times = zeros(size(C));
models = {};
for i = 1:length(gama) % different gama
    parfor j = 1:length(C) % different C
        % C-SVC, radial basis function
        model = svmtrain(YTrain,XTrain,...
                ['-s 0 -t 1 -c ', convertStringsToChars(string(C(i))),...
                 ' -g ', convertStringsToChars(string(gama(j)))]);
        [predicted_label, accuracy, dec_values_L] = svmpredict(YTest, XTest,model);
        accuracies(i,j) = accuracy(1);
        models{i,j} = model;  
    end
end
server_chan("Simulation Done")