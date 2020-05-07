
train_testAcc = [];

for numSim = 1:1

clearvars -except numSim train_testAcc ; clc;
%% This is the demo for MNIST dataset.
rand('state',0)

cd train_testMNIST/
matFileTrain = sprintf('MNIST_train_%d.mat',numSim);
matFileTest = sprintf('MNIST_test_%d.mat',numSim);
load(matFileTrain)
load(matFileTest)

cd ..
cd basisVec/
basisVec = sprintf('basisVec_MNIST_%d.mat',numSim);
load(basisVec);

cd ..
train_x = train_MNIST;
train_y = double(labels_train);
test_x = test_MNIST;
test_y = double(labels_test);


%% Randomness

usefulInd = find(sum(s2Matrix)<150);

N1 = length(usefulInd);
N2 = 10;

% Hard-coded pooling
pos = datasample(1:size(synapticMatrix,2),N1,'Replace',false);
b1_hc = synapticMatrix(:,pos);
b1_hc = [b1_hc; rand(N1,1)'];

% Self organized pooling
b1_selfOrg = s2Matrix(:,usefulInd);
b1_selfOrg = [b1_selfOrg; rand(N1,1)'];

% Randomized 
pos = datasample(1:size(randomSynMatrix,2),N1,'Replace',false);
b1_rand = randomSynMatrix(:,pos);
b1_rand = [b1_rand; rand(N1,1)'];

b2=2*rand(N1+1,N2)-1;
b3 = [];

%% If the RAM of your computer is less than 16G, you may try the following version which has less than half of the hidden nodes
 C = 2^-30 ;s = 1;
% load random_700_700_5000.mat;
%% Call the training function
[Training_hc, Testing_hc, Training_time, Testing_time] = helm_train_modify2(double(train_x), double(train_y), double(test_x), double(test_y), b1_hc, b2, b3, s, C);
[Training_selfOrg, Testing_selfOrg, Training_time, Testing_time] = helm_train_modify2(double(train_x), double(train_y), double(test_x), double(test_y), b1_selfOrg, b2, b3, s, C);
[Training_rand, Testing_rand, Training_time, Testing_time] = helm_train_modify2(double(train_x), double(train_y), double(test_x), double(test_y), b1_rand, b2, b3, s, C);

train_testAcc = [train_testAcc; [Training_hc*100, Testing_hc*100,Training_selfOrg*100, Testing_selfOrg*100,Training_rand*100, Testing_rand*100]];

end