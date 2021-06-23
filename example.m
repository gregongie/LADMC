% Example usage of LADMC and iLADMC code
% Greg Ongie, June 23, 2021
% gregory.ongie@marquette.edu 
%% generate unions of subspaces data
rng(1);  %fix random seed for reproducibility
n = 15;  %ambient dim
r = 3;   %subspace dim
k = 5;   %no. of subspaces
p = 50;  %no. points per subspace
m = 12;  %no. samples per column
X = [];
for i=1:k
    U = orth(randn(n,r));
    X = [X, U*randn(r,p)];
end
s = size(X,2);
Xtrue = X;
R = rank(phi2hom(Xtrue),1e-5); %rank of tensorized matrix
errfun = @(X) norm(X-Xtrue,'fro')/norm(Xtrue,'fro'); %error metric
%% subsample data
%sample m entries of each column uniformly at random
sampmask = false(n,s);
for j = 1:s
     randind = randperm(n,m);
     sampmask(randind,j) = true;
end     
samples = Xtrue(sampmask);
Xinit = zeros(n,s);
Xinit(sampmask) = samples;
%% run LADMC
niter = 1000; %# SVD iterative hard thresholding iterations
[X,err] = ladmc2(Xinit,sampmask,samples,R,niter,errfun);
fprintf('LADMC:\t Final NRMSE=%1.2e\n',errfun(X));
%% run iLADMC - iterative version of LADMC
% Iteratively apply LADMC with early stopping of matrix completion step
% Typically reaches lower error in fewer iterations than plain LADMC
nouter = 50; %# of outer loop LADMC iterations
ninner = 20; %# of inner loop SVD iterative hard thresholding iterations
[X,err] = iladmc2(Xinit,sampmask,samples,R,ninner,nouter,errfun);
fprintf('iLADMC:\t Final NRMSE=%1.2e\n',errfun(X));
figure(1); semilogy(err); xlabel('outer iteration'); ylabel('NRMSE');
