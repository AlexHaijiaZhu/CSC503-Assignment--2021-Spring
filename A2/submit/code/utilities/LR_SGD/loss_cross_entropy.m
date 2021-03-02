function loss = loss_cross_entropy(y,y_hat) 
    loss = -y*log(y_hat)-(1-y)*log(1-y_hat);
end
