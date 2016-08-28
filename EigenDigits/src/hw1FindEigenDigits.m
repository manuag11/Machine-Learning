function [v_normalized mu] = hw2FindEigenDigits(A)

m=28;
n=28;
[x k]=size(A);

%top_k represents the number of eigenvectors we want to take for training
%purposes
top_k=k;

%subtract the mean of features from A
mean_vec=mean(A,2);
A=double(A);
A_normalized=bsxfun(@minus,A,mean_vec);

%calculate the eigenvalues and eigenvectors of A_normalized'A_normalized
[V mu]=eig(A_normalized'*A_normalized);

%calculate the eigenvectors of Sigma from those of A'A
v_original=A_normalized*V;

%sort eigenvectors in descending order of eigenvalues
sum_vector=sum(mu);
[sum_vector index]=sort(sum_vector);
v_sorted=v_original(:,index(end:-1:1));

%extract top_k eigenvectors
v_sorted=v_sorted(:,1:top_k);

%normalizing the eigenvectors such that their norm becomes one
[dim1 dim2]=size(v_sorted);
v_normalized=zeros(dim1,dim2);
for i=1:dim2
    v_normalized(:,i)=v_sorted(:,i)/norm(v_sorted(:,i));
end
end