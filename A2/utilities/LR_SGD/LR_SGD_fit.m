function [w,b,loss,lr] = LR_SGD_fit(w, b, X, Y, LR)

global t alpha t0 d r C learning_rate_init
no_iterations = length(X(:,1)); % number of iterataion
loss = zeros([1,no_iterations]);
lr = zeros([1,no_iterations]);
    for i =1:no_iterations
        feature = X(i,:);
        label = Y(i,:);
 %% predicted result
        hyber = dot(w,feature)+b;
        Y_hat = my_sigmod(hyber); 
        cost = loss_cross_entropy(label, Y_hat) + 1/2*(norm(w))^2;

%% Gradient calculation
        dw = C*(Y_hat-label) .*feature + w;
        db = C*(Y_hat-label) * 1;
        
%% update learn rate
        switch LR
            case 2
                learning_rate = learning_rate_init * d^floor((1+i/r));
            case 1
                learning_rate = alpha/(t0+t);t = t+1/100;
            otherwise
                learning_rate = LR;
        end
        %% update parameter 
        w = w - learning_rate.*dw;
        b = b - learning_rate.*db;
        %% peak how the wight are updated
        if mod(i,1001)==0
            figure(1)
            hold on 
            plot(w)
        end
        
        loss(i)=cost;
        lr(i) = learning_rate;
    end
    learning_rate_init = learning_rate;
end

