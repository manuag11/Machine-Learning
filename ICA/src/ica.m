function Y = ica(X,max_iterations,eta,m,n,delta_W_threshold)
%	SUMMARY performs ICA on the mixed signals and returns the source signals thus
%	obtained
%   DESCRIPTION U is the matrix of source signals, A is the mixing matrix
%   and Y is the approximation to source signals found by the ICA algorithm

%W:nxm
W=rand(n,m).*0.1;
t=size(X,2);
num_iterations=0;
delta_W_norm=1;
%run this loop till we have not reached the maximum number of iterations
%and the norm of delta_W is greater than the threshold
while((num_iterations<max_iterations)&&(delta_W_norm>delta_W_threshold))
    %Y:nxt
    Y=W*X;
    Z=Y;
    for i=1:n
        Z(i,:)=sigmf(Z(i,:),[1 0]);
    end
    delta_W=eta*(eye(n)+(ones(n,t)-2.*Z)*Y')*W;
    W=W+delta_W;
    delta_W_norm=norm(delta_W);
    num_iterations=num_iterations+1;
end
Y=W*X;
end

