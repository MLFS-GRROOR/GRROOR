function  [RM]= GloRed_eva(X,H,d)
for i = 1:d
    f_i = H*X(i,:)';
    F(:,i) = f_i;
end

for i = 1 : d
    for j = 1 : d
     B(i,j) = F(:,i)'* F(:,j)/(norm(F(:,i))*norm(F(:,j)));
end
end

RM = B.*B;