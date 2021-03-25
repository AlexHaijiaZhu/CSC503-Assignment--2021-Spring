function centorid = k_means_init(X, k,initializer)
[d,n] = size(X);
if d<k
    disp('Number of cluster should smaller than the number of input examples!')
end
centorid = zeros(k,n);

if initializer == 2 % k-means ++
    unique_examples = unique(X,'rows');
    [x,y] = size(unique_examples);
    % pick the first center at uniform rand
    idx_c_before = round(rand(1)*x);
    centorid(1,:) = unique_examples(idx_c_before,:);
    X_remain = unique_examples(1:x~=idx_c_before,:);
    % pick the rest of the centers
    for i = 2:k
        % calauate the distance
        clear D
        for j = 1:i-1
            temp = X_remain-centorid(j,:);
            D(j,:) = sqrt(dot(temp,temp,2));
        end
        [M,I] = min(D,[],1);
        % example with farthest 
        pr = M/sum(M);
        [M,I] = max(pr);
        centorid(i,:) = X_remain(I,:);
        % updat remaining X
        X_remain = X_remain(1:(x-i+1)~=I,:);

    end
    
    disp("Initializer: k-means ++")

else % random initialization
%     centorid = rand(k,n) .* (max(max(X))-min(min(X)));
    unique_examples = unique(X,'rows');
    [x,y] = size(unique_examples);
%     for i=1:k
%         centorid(i,:) = unique_examples(round(rand(1)*x),:);
%     end
    idxs = randsample(x,k);
    centorid = unique_examples(idxs,:);
    disp("Initializer: random")
end
end % end of init