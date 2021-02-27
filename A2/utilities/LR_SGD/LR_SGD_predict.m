%% --------------------------------------------------
% Predict

function [label,prob,accuracy] = LR_SGD_predict(w,b,X,Y)
prob=ones([length(X),1]);
for i = 1: length(prob)
    prob(i) = my_sigmod(dot(w,X(i,:))+b);    
end
% figure
% plot(label, '*')
label(prob>0.5,1)=1;
label(prob<0.5,1)=0;

if sum(prob(prob==0.5))>0
    disp("find some label in the middle")
end

accuracy=sum(label==Y)./length(Y)*100;
end
