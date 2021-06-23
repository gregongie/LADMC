function X = preimage_svd(X2,sampmask,samples,n)
%PREIMAGE_SVD Obtain rank 1 SVD decomposition of tensorized matrix
%X2         tensorized matrix, same dimensions and indexing as output of phi2hom(X)
%sampmask   binary mask of sampling locations
%samples    vectors of samples at sampmask locations
%n          row dimension of output X

s = size(X2,2);
trimask = logical(triu(ones(n),0));
X0 = zeros(n,s);
X0(sampmask) = samples;
X = zeros(n,s);
for j=1:s %each column do (note: this loop could be parallized)
    x2 = X2(:,j);
    xx = zeros(n);
    xx(trimask) = x2;  %upper triangular matrix
    xx = xx + xx' - diag(diag(xx)); %make symmetric n-by-n matrix
    [U,S,~] = svd(xx); %note: svd typically faster than eig or eigs
    evec = U(:,1);
    eval = S(1);
    x = sqrt(abs(eval))*evec; %x up to sign
    [~,ind] = max(X0(:,j)); %find largest non-zero value
    if sign(x(ind)) ~= sign(X0(ind,j)) %compare sign at non-zero value
        x = -x; %flip sign if it is wrong
    end
    X(:,j) = x;
end
end

