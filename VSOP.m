% Joint Visual and Semantic Optimization for Zero-shot Learning
% Knowledge-Based Systems, 2021
% Authors: Hanrui Wu, Yuguang Yan, Sentao Chen, Xiangkang Huang, Qingyao Wu, Michael K. Ng
% Code written by Hanrui Wu, 
% based on the code ManOpt provided by Z. Wen and W. Yin (reference: A feasible method for optimization with orthogonality constraints)


function [acc_list, predict_labels] = VSOP(Xs, Zs, Xu, Zu, Yu, options)

V = size(Xs, 2);
S = size(Zs, 2);

lambda_1 = options.lambda_1;
lambda_2 = options.lambda_2;
gamma_1 = options.gamma_1;
gamma_2 = options.gamma_2;
max_iter = options.max_iter;


A = (1 - gamma_1) * (Xs' * Xs) + lambda_1 * eye(V, V);
B = (1 - gamma_2) * (Zs' * Zs) + lambda_2 * eye(S, S);
C = Xs' * Zs;


P = options.P;
Q = options.Q;

opts.record = 0; %
opts.mxitr  = 1000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

acc_list = [];
for t = 1:max_iter
    
    fprintf('%d trial, ', t);
    %%%%%%%%% fix P and update Q %%%%%%%%%
    [new_Q, ~, ~]= OptStiefelGBB(Q, @funQ, opts, B, C, P);
    
    %%%%%%%%% fix Q and update P %%%%%%%%%
    [new_P, ~, ~]= OptStiefelGBB(P, @funP, opts, A, C, new_Q);
    
    tilde_Xu = Xu * new_P;
    tilde_Zu = Zu * new_Q;
    
    predict_labels = knnclassify(tilde_Xu, tilde_Zu, Yu, 1);
    per_acc = computeAcc(predict_labels, Yu, unique(Yu)) * 100;
    fprintf('acc_per_class = %.1f\n', per_acc);
    
    acc_list = [acc_list per_acc];
    
    P = new_P;
    Q = new_Q;
    
end

end

function [F, G] = funQ(X, B, C, P)
G = 2 * B * X - 2 * C' * P;
F = trace(X' * B * X) - 2 * trace(P' * C * X);
end

function [F, G] = funP(X, A, C, Q)
G = 2 * A * X - 2 * C * Q;
F = trace(X' * A * X) - 2 * trace(C * Q * X');
end





