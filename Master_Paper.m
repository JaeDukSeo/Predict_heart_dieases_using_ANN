%% 
% This m file tried to replicate the mastet thesis paper from 
% Original Author of the Code : Jae Duk Seo 
% Huge thanks to He for sharing the knowledge and 
% data. It was a experience for me to more deeply 
% understand machine learning (specificaly ANN)
%% Initialization
clear ; close all; clc
%% Part 1 - Read csv file and randomly disturbute them
M = csvread('ImputedDatabase.csv');
y  = M(:,22);
M = M(:,3:21);
[m,n] = size(M);
% Seperate the training set | crossvalidation | test set
trainset_num = (m *  0.7375); % 59 person 
trainset = zeros(trainset_num,n);
y_train = zeros(trainset_num,1);
cross_num =     (m * 0.0); % 0 percent - will do this later
cross     = zeros(cross_num,n);
y_cross = zeros(cross_num,1);
test_num =       (m * 0.2625); % 21 person
test        = zeros(test_num,n);
y_test = zeros(test_num,1);
% Generate Random number to be assigned
num_array = randperm(m,m);
for i = 1:trainset_num
    trainset(i,:) = M(num_array(i),:);
    y_train(i) = y(num_array(i));
end
for i = 1:cross_num
    cross(i,:) = M(num_array(i+trainset_num),:);
    y_cross(i) = y(num_array(i+trainset_num));
end
for i = 1:test_num
    test(i,:) = M(num_array(i+trainset_num+cross_num),:);
    y_test(i) = y(num_array(i+trainset_num+cross_num));
end
clearvars M y i

% Setting the Parameter of the datas
input_layer_size  = 19;  % Number of features
hidden_layer_size = 40;   % 25 hidden units
num_labels = 2;          % We have either 0/1
fprintf('Declared all of the variables.\n');
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part2 - Forward Propagation of NN without Regularization 
% Initilize the two Theta values 
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
% Weight regularization parameter (we set this to 0 here).
lambda = 0;
X = trainset;
y = y_train;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters : %f \n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part 3 - Forward Propagation with Regularzation 
lambda = 0.000001;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['Cost at parameters : %f \n'], J);
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part 4 - Train the ANN 
fprintf('\nTraining Neural Network... \n')
% Set the interation value - more larger = more accurate
options = optimset('MaxIter',300);
lambda = 0;
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
 [nn_params, cost] = fmincg(costFunction, nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('\nNew Theta 1 and Theta 2 have been obtained\n')
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part 5 - Testing the accuracy for the Training set
act_one = [ones(size(X,1),1) X];
z_two = act_one * Theta1';
act_two = sigmoid(z_two);
act_two = [ones(size(act_two,1),1) act_two];
z_three = act_two * Theta2';
act_three = sigmoid(z_three);
h_theta = act_three;
[temp,check] = max(h_theta,[],2);
check = check-1;
error = 0;
for i = 1:size(check,1)
    if y(i) ~= check(i)
        error = error +1;
    end
end
fprintf('\nThe error rate for the training set is : %f percent \n',(error/size(check,1))*100)
fprintf('Program paused. Press enter to continue.\n');
pause;
%% Part 6 - Testint the accuracy for the Test set 
X = test;
y =y_test;
act_one = [ones(size(X,1),1) X];
z_two = act_one * Theta1';
act_two = sigmoid(z_two);
act_two = [ones(size(act_two,1),1) act_two];
z_three = act_two * Theta2';
act_three = sigmoid(z_three);
h_theta = act_three;
[temp,check] = max(h_theta,[],2);
check = check-1;
error = 0;
for i = 1:size(check,1)
    if y(i) ~= check(i)
        error = error +1;
    end
end
fprintf('\nThe error rate for the test set is : %f percent \n',(error/size(check,1))*100)
fprintf('Program paused. Press enter to continue.\n');
pause;