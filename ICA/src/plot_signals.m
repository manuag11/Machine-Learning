function plot_signals(U,X,Y)
%	SUMMARY plots two signal matrices U and Y

%color matrix for plotting signals with different colors
colors=['k','b','r','g','y','m','c'];

figure('name','Original Signals')
[n t]=size(U);
%random permutation for coloring signals randomly
random_permutation=randperm(7);
for i=1:n
subplot(n,1,i);
plot(U(i,:),'color',colors(random_permutation(mod(i,7))));
end

m=size(X,1);
figure('name','Mixed Signals')
%random permutation for coloring signals randomly
random_permutation=randperm(7);
for i=1:m
subplot(m,1,i);
plot(X(i,:),'color',colors(random_permutation(mod(i,7))));
end

figure('name','Recovered Signals')
%random permutation for coloring signals randomly
random_permutation=randperm(7);
for i=1:n
subplot(n,1,i);
plot(Y(i,:),'color',colors(random_permutation(mod(i,7))));
end
end

