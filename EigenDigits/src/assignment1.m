%loading the data into the workspace
load('digits.mat');

%getting the size of trainImages and storing the dimensionality of the
%image in variables m and n. In other words, each image is an mxn image.
[m n o samples]=size(trainImages);

%generating a random permutation for training images selection
permutation=randperm(60000);

k=500;
%k represents the number of samples we use for training
%randomly select k images for training purposes

temp=trainImages(:,:,1,permutation(1:k));
train_labels=trainLabels(permutation(1:k));
%imshow(temp(:,:,:,1));
%pause

%reshape the training images vector
A=reshape(temp,m*n,k);

%calling the function hw1Eigendigits that returns the normalized vector of
%eigen vectors sorted in descending order of eigen values. The matrix mu is
%the diagonal matrix of the eigen values
display('Extracting eigenvectors from the training samples...');
[v_normalized mu]=hw1FindEigenDigits(A);

display('Projecting the training data onto the eigenspace...');
%subtracting the mean of the X's from the matrix A
mean_vec=mean(A,2);
A=double(A);
A_normalized=bsxfun(@minus,A,mean_vec);

%projecting the training data onto the eigen space for training
z=v_normalized'*A_normalized;

display('Preparing to display reconstructed training digits...');
display('Hit enter to move to the next digit. A total of 5 training digits will be displayed.');
%displaying the reconstructed digits
z_train_imshow=v_normalized*z;
z_reshaped=reshape(z_train_imshow,m,n,k);
for i=1:5
    imshow(z_reshaped(:,:,i));
    pause
end

%test_size denotes the number of images we want to test
test_size=1000;

%permutation for test images
permutation_test=randperm(10000);

display('Projecting the test data onto the eigenspace...');
%extracting test images randomly and reshaping them
test_img=testImages(:,:,1,permutation_test(1:test_size));
test_labels=testLabels(permutation_test(1:test_size));
test_A=reshape(test_img,m*n,test_size);

%projecting the test data onto the eigen space
test_A=double(test_A);
test_A_normalized=bsxfun(@minus,test_A,mean_vec);
z_test=v_normalized'*test_A_normalized;

display('Preparing to display reconstructed test digits...');
display('Hit enter to move to the next digit. A total of 5 test digits will be displayed.');
z_test_imshow=v_normalized*z_test;
z_test_reshaped=reshape(z_test_imshow,m,n,test_size);
for i=1:5
    imshow(z_test_reshaped(:,:,i));
    pause
end

A=double(A);
train_labels=double(train_labels);
z_tested=double(z_test);

display('Running k-nearest neighbors classification algorithm...');
%application of k-nearest neighbor algorithm to get the test output labels
test_output=knnclassify(z_test',z',train_labels',1,'euclidean');
test_output=test_output';

display('Calculating accuracy...');
%calculating accuracy
accuracy=0;
for i=1:test_size
    if(test_output(i)==test_labels(i))
      accuracy=accuracy+1;  
    end
end
accuracy=accuracy*100/test_size;
fprintf('\nAccuracy = %f\n',accuracy);