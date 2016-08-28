function [DAG,A,B,C,D,E] = constructDAG(N)
% This function creates and returns a DAG with N nodes
DAG = zeros(N,N);
A = 1; B = 2; C = 3; D = 4; E = 5;
DAG(A,C) = 1;
DAG(B,C) = 1;
DAG(C,D)=1;
DAG(C,E)=1;

end

