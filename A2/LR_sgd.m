

% random initial point:w and b
w=rand(size(XTrain));
b= rand(1);
learning_rate = 0.001;
no_iterations = 1000;
for i=1:length(XTrain)
    LR_re_predict(w, b, XTrain(i,:), YTrain(i),learning_rate, no_iterations)
end

