DEBUG=1;
%DEBUG = 1 represents loading sample data from icaTest.mat
%DEBUG = 0 represents loading sound data from sounds.mat
if(DEBUG)
    load('icaTest.mat');
else
    load('sounds.mat');
    %taking sound signals 1,3 and 4
    U=sounds([1 3 4],:);
end
[n t]=size(U);
%m represents the number of samples
%n represents the number of source signals
%t represents the number of time points in signals
m=n;
max_iterations=100000;
delta_W_threshold=10^-15;
eta=0.1;
%calculating the mixture of signals
if(DEBUG)
    X=A*U;
else
    X=mix_signals(U,m);
end
%applying ICA
Y=ica(X,max_iterations,eta,m,n,delta_W_threshold);
%plotting the retrieved signals
plot_signals(rescale(U,1,0),rescale(X,1,0),rescale(Y,1,0));