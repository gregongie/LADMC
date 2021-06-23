function [X,err] = ladmc2(Xinit,sampmask,samples,R,niter,errfun)
%LADMC2 Low-algebraic dimension matrix completion, 2nd order
%Xinit     initialization
%sampmask  binary mask of sampling locations
%samples   vector of samples
%R         rank cutoff in tensor space
%ninner    number of SVD hard thresholding iterations
%errfun    function handle to compute error metric

X2 = phi2hom(Xinit);
sampmask2 = phi2hom(double(sampmask))~=0;
samples2 = X2(sampmask2);
for i=1:niter
    %SVD hard thresholding
    [U,S,V] = svd(X2,'econ');
    s = diag(S);
    X2 = U(:,1:R)*diag(s(1:R))*V(:,1:R)';
    X2(sampmask2) = samples2;
end
X = preimage_svd(X2,sampmask,samples,size(Xinit,1));
X(sampmask) = samples;
err = errfun(X);

end