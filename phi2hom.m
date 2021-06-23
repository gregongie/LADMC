function phiX = phi2hom(X)
%PHI2 - create exact homogeneous polynomial lifting phi_2(X)
[n,s] = size(X);
phiX = zeros(nchoosek(n+1,2),s);
trimask = logical(triu(ones(n),0)); %upper triagular mask
for j = 1:s
    x = X(:,j);
    x2 = x*x';
    phiX(:,j) = x2(trimask);
end

end

