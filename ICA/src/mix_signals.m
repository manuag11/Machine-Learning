function X = mix_signals(U,m)
%   SUMMARY mixes signals using a randomly generated matrix A
[n t]=size(U);
A=rand(m,n);
%X:mxt; A:mxn; U:nxt
X=A*U;
end

