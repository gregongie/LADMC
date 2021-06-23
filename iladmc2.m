function [X,err] = iladmc2(Xinit,sampmask,samples,R,ninner,nouter,errfun)
%ILADMC2 Iterative low-algebraic dimension matrix completion, 2nd order
%Xinit     initialization
%sampmask  binary mask of sampling locations
%samples   vector of samples
%R         rank cutoff in tensor space
%ninner    number of ``inner'' SVD hard thresholding iterations
%nouter    number of ``outer'' pre-image iterations
%errfun    function handle to compute error metric

n = size(Xinit,1);

%generate samples in lifted domain
X = Xinit;
X2 = phi2hom(Xinit);
sampmask2 = phi2hom(double(sampmask))~=0;
samples2 = X2(sampmask2);

err = [];
for i=1:nouter
    X2 = phi2hom(X);
    for j=1:ninner
        [U,S,V] = svd(X2,'econ'); s = diag(S);
        X2 = U(:,1:R)*diag(s(1:R))*V(:,1:R)';
        X2(sampmask2) = samples2;
    end
    X = preimage_svd(X2,sampmask,samples,n);   
    X(sampmask) = samples;
    err(end+1) = errfun(X);
end
end