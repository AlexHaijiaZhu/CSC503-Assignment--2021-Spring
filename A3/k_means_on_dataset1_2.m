X = dataset1;
k = 3;
no_iter = 10;
initializer = 1;

[c_idx,C] = mykmeans(X,k,no_iter,initializer);
C

initializer = 2;
[c_idx,C] = mykmeans(X,k,no_iter,initializer);