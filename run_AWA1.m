% Demo for VSOP method on the AWA1 data set

clc;
clear;

addpath('vsop/');

load(['./data/AWA1/res101.mat']);
load(['./data/AWA1/att_splits.mat']);

X = features';
Y = labels;
Z = att;

%% preprocess
X = X';
meanX = mean(X, 2);
X = bsxfun(@minus, X, meanX);
X = bsxfun(@times, X, 1./max(1e-12, sqrt(sum(X.^2))));
X = X';

meanZ = mean(Z, 2);
Z = bsxfun(@minus, Z, meanZ);
Z = bsxfun(@times, Z, 1./max(1e-12, sqrt(sum(Z.^2))));

Z = Z(:, Y);
Z = Z';


Xs = X(trainval_loc, :);
Ys = Y(trainval_loc);
Zs = Z(trainval_loc, :);

Xts = X(test_unseen_loc, :);
Yts = Y(test_unseen_loc);
Zts = Z(test_unseen_loc, :);

Xtr = X(test_seen_loc, :);
Ytr = Y(test_seen_loc);
Ztr = Z(test_seen_loc, :);


fprintf(1, 'running dataset: AWA1 \n');

%%%%%%%%% parameter %%%%%%%%%
load('AWA1_param.mat');
options.lambda_1 = lambda_1;
options.lambda_2 = lambda_2;
options.gamma_1 = gamma_1;
options.gamma_2 = gamma_2;
options.D = D;
options.max_iter = 10;
options.P = P;
options.Q = Q;


[acc_list, predict_labels] = VSOP(Xs, Zs, Xts, Zts, Yts, options);


rmpath('vsop/');










