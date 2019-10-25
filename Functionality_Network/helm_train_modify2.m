function [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = helm_train(train_x,train_y,test_x,test_y,b1,b2,b3,s,C)
tic

%b1 = b1_hc;

train_x = zscore(train_x')';
H1 = [train_x .1 * ones(size(train_x,1),1)];
clear train_x;
%% First layer RELM
A1 = H1 * b1;A1 = mapminmax(A1);
beta1 = b1;
clear b1;
%beta1  =  sparse_elm_autoencoder(A1,H1,1e-3,50)';
clear A1;

T1 = H1 * beta1;
fprintf(1,'Layer 1: Max Val of Output %f Min Val %f\n',max(T1(:)),min(T1(:)));

[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';

clear H1;

l1 = max(max(T1));l1 = s/l1;
fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',l1,min(T1(:)));

T1 = tansig(T1 * l1);

beta = (T1'  *  T1+eye(size(T1',1)) * (C)) \ ( T1'  *  train_y);
Training_time = toc;
disp('Training has been finished!');
disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%% Second layer RELM
% H2 = [T1 .1 * ones(size(T1,1),1)];
% clear T1;
% 
% A2 = H2 * b2;A2 = mapminmax(A2);
% clear b2;
% beta2 = sparse_elm_autoencoder(A2,H2,1e-3,50)';
% clear A2;
% 
% T2 = H2 * beta2;
% fprintf(1,'Layer 2: Max Val of Output %f Min Val %f\n',max(T2(:)),min(T2(:)));
% 
% [T2,ps2] = mapminmax(T2',0,1);T2 = T2';
% 
% clear H2;
%% Original ELM
% H2 = [T1 .1 * ones(size(T1,1),1)];
% clear T2;
% 
% T2 = H2 * b2;
% l2 = max(max(T2));l2 = s/l2;
% fprintf(1,'Layer 3: Max Val of Output %f Min Val %f\n',l2,min(T2(:)));
% 
% T2 = tansig(T2 * l2);
% clear H2;
%% Finsh Training
% beta = (T2'  *  T2+eye(size(T2',1)) * (C)) \ ( T2'  *  train_y);
% Training_time = toc;
% disp('Training has been finished!');
% disp(['The Total Training Time is : ', num2str(Training_time), ' seconds' ]);
%% Calculate the training accuracy
xx = T1 * beta;
clear T3;

yy = result_tra(xx);
train_yy = result_tra(train_y);
TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);
disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);
%% First layer feedforward
tic;

test_x = zscore(test_x')';
HH1 = [test_x .1 * ones(size(test_x,1),1)];
clear test_x;

TT1 = HH1 * beta1; TT1  =  mapminmax('apply',TT1',ps1)';
clear HH1;clear beta1;

TT1 = tansig(TT1*l1);


%% Second layer feedforward
%HH2 = [TT1 .1 * ones(size(TT1,1),1)];
%clear TT1;

%TT2  =  HH2 * beta2;TT2  =  mapminmax('apply',TT2',ps2)';
%clear HH2;clear beta2;
%% Last layer feedforward
% HH2 = [TT1 .1 * ones(size(TT1,1),1)];
% clear TT1;
% 
% TT2 = tansig(HH2 * b2 * l2);
% clear HH3;clear b3;

x = TT1 * beta;
y = result_tra(x);
test_yy = result_tra(test_y);
TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);
clear TT2;
%% Calculate the testing accuracy
Testing_time = toc;
disp('Testing has been finished!');
disp(['The Total Testing Time is : ', num2str(Testing_time), ' seconds' ]);
disp(['Testing Accuracy is : ', num2str(TestingAccuracy * 100), ' %' ]);
