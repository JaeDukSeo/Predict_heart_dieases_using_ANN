%% 
% This m file applys the NN of most accurate value of Thetas
% Original Author of the Code : Jae Duk Seo 
% Huge thanks to He for sharing the knowledge and 
% data. It was a experience for me to more deeply 
% understand machine learning (specificaly ANN)
%% Initialization
clear ; close all; clc

%% Part 1 - Read the variables
load('Accurate_Train.mat');
fprintf('\nNow you should see two variables one called trainset and');
fprintf('\nthe other called ytrain. Trainset is the set which includes all of the');
fprintf('\nfeatures to include and y_train is the present of Ischaemia0or1 ');
pause;
load('Accurate_Test.mat');
fprintf('\n\nNow you should see two variables one called testset and');
fprintf('\nthe other called y_test. Testset is the set which includes all of the');
fprintf('\nfeatures to test and y_test is the present of Ischaemia0or1 to calculate accuracy');
pause;
load('Accurate_Theta.mat');
fprintf('\n\nThese are the two obtained Theta values (eg weights) for the train/test set.\n')

%% Part 2 - Forward Propagation on the Training Set
X = trainset;
y = y_train;
[m,n] = size(X);
activate_one = [ones(m,1) X];
z_two  = activate_one * Theta1';
activate_two = sigmoid(z_two);
activate_two = [ones(m,1) activate_two];
z_three = activate_two * Theta2';
activate_three =  sigmoid(z_three);
htheta = activate_three; % This is the predicted value
[temp,check] = max(htheta,[],2);
check = check -1; % Since our values are either 0/1
error = 0 ;
% Loop through all of the predicted value and ground_truth, and if they are
% different increment the error value.
for i = 1:size(check,1)
    if y(i) ~= check(i)
        error = error + 1;
    end
end
fprintf('\nError rate for the training set :  %f \n',error)
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part 3 - Forward Propagation on the Test Set
X = test;
y = y_test;
[m,n] = size(X);
activate_one = [ones(m,1) X];
z_two  = activate_one * Theta1';
activate_two = sigmoid(z_two);
activate_two = [ones(m,1) activate_two];
z_three = activate_two * Theta2';
activate_three =  sigmoid(z_three);
htheta = activate_three; % This is the predicted value
[temp,check] = max(htheta,[],2);
check = check -1; % Since our values are either 0/1
error = 0 ;
% Loop through all of the predicted value and ground_truth, and if they are
% different increment the error value.
for i = 1:size(check,1)
    if y(i) ~= check(i)
        error = error + 1;
    end
end
fprintf('\nError rate for the Test set :  %f \n',error)
fprintf('Program paused. Press enter to continue.\n');
pause;