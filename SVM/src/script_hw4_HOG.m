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

cellSize = [4 4];
temp2=reshape(temp(:,:,:,1),[28 28]);
temp2=double(temp2);
[hog_4x4, vis4x4] = extractHOGFeatures(temp2, 'CellSize', cellSize);
hogFeatureSize = length(hog_4x4);

training_features  = zeros(k, hogFeatureSize, 'single');
for i=1:k
    temp2=reshape(temp(:,:,:,i),[28 28]);
    temp2=double(temp2);
    training_features(i, :) = extractHOGFeatures(temp2, 'CellSize', cellSize);
end
%reshape the training images vector to generate training_features
training_features=double(training_features);
display(size(training_features))
display(size(train_labels))

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(training_features, train_labels');

%generating a random permutation for training images selection
permutation=randperm(5000);

temp1=testImages(:,:,1,permutation(1:test_samples));
test_labels=testLabels(permutation(1:test_samples));

testing_features  = zeros(test_samples, hogFeatureSize, 'single');
for i=1:k
    temp2=reshape(temp1(:,:,:,i),[28 28]);
    temp2=double(temp2);
    testing_features(i, :) = extractHOGFeatures(temp2, 'CellSize', cellSize);
end

testing_features=double(testing_features);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testing_features);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(test_labels, predictedLabels)
conf_size=size(confMat,2);
accuracy=0;
for i = 1:conf_size
    accuracy=accuracy+confMat(i,i);
end
accuracy=accuracy/test_samples;
display(accuracy);