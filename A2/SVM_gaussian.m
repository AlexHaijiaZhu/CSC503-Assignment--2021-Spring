% Fourth: Gaussian kernel
C = [1e-5, 1e-3, 1e-2, 1e-1, eps, 1, 1e1, 1e2, 1e3];
gama = [1e-5, 1e-3, 1e-2, 1e-1, eps, 1, 1e1, 1e2, 1e3];
accuracies = zeros(length(l),length(C));
traing_times = zeros(size(C));
models = {};
for i = length(l) % different gama
    parfor j = 1:length(C) % different C
        % C-SVC, radial basis function
        model = svmtrain(YTrain,XTrain,...
                ['-s 0 -t 1 -c ', convertStringsToChars(string(C(i))),...
                 ' -g ', convertStringsToChars(string(gama(jj)))]);
        [predicted_label, accuracy, dec_values_L] = svmpredict(YTest, XTest,model);
        accuracies(i,j) = accuracy(1);
        models{i,j} = model;  
    end
end