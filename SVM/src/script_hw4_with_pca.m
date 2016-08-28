%loading the data into the workspace
load('digits.mat');

%getting the size of trainImages and storing the dimensionality of the
%image in variables m and n. In other words, each image is an mxn image.
[m n o samples]=size(trainImages);

%generating a random permutation for training images selection
permutation=randperm(60000);

k=500;
%k represents the number of samples we use for training
test_samples=500;

%randomly select k images for training purposes
temp=trainImages(:,:,1,permutation(1:k));
train_labels=trainLabels(permutation(1:k));
%imshow(temp(:,:,:,1));
%pause

%reshape the training images vector to generate training_features
training_features=reshape(temp,m*n,k);
training_features=double(training_features);
%display(size(training_features))
%display(size(train_labels))

%calling the function hw1Eigendigits that returns the normalized vector of
%eigen vectors sorted in descending order of eigen values. The matrix mu is
%the diagonal matrix of the eigen values
display('Extracting eigenvectors from the training samples...');
[v_normalized mu]=hw1FindEigenDigits(training_features);

display('Projecting the training data onto the eigenspace...');
%subtracting the mean of the X's from the matrix A
mean_vec=mean(training_features,2);
training_features=double(training_features);
training_features_normalized=bsxfun(@minus,training_features,mean_vec);

%projecting the training data onto the eigen space for training
projected_training_features=v_normalized'*training_features_normalized;
display(size(projected_training_features))

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
%params = templateSVM('KernelFunction', 'linear', 'KernelScale', 'auto');
%classifier = fitcecoc(projected_training_features', train_labels', 'Learners', params,'Coding', 'onevsall');
classifier = fitcecoc(projected_training_features', train_labels');

%generating a random permutation for training images selection
permutation=randperm(5000);

temp1=testImages(:,:,1,5000+permutation(1:test_samples));
test_labels=testLabels(5000+permutation(1:test_samples));
testing_features=reshape(temp1,m*n,test_samples);
%testing_features=double(testing_features);

%projecting the test data onto the eigen space
test_features=double(testing_features);
test_features_normalized=bsxfun(@minus,test_features,mean_vec);
projected_test_features=v_normalized'*test_features_normalized;

% Make class predictions using the test features.
predictedLabels = predict(classifier, projected_test_features');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(test_labels, predictedLabels)
conf_size=size(confMat,2);
accuracy=0;
for i = 1:conf_size
    accuracy=accuracy+confMat(i,i);
end
accuracy=accuracy/test_samples;
display(accuracy);