close all;
clear all;
clc;

addpath('data');
addpath('tools');

load('Isolet1.mat');

X = fea';
gnd = gnd';
K = max(gnd);
n = size(X, 2);

for idx = 1 : n
    X(:, idx) = X(:, idx) ./ max(1e-12,  norm(X(:, idx)));
end

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

lamdas = [20];
aplas = [0.3];

%lamdas = [10, 15, 20, 25, 30];
%aplas = [0.1, 0.2, 0.3, 0.4, 0.5];
apla_len = length(aplas);
lmd_len = length(lamdas);

D = calculate_similarity(X');
XtX = X' * X;
for lmd_idx = 1 : lmd_len
    lambda = lamdas(lmd_idx);
    Z = (XtX + lambda * eye(n)) \ XtX;
    for i = 1 : n
        Z(:, i) = Z(:, i) ./ max(1e-12, sum(Z(:, i)));
    end
    for apla_idx = 1 : apla_len
        apla = aplas(apla_idx);      
        W = Z;
        tic;
        T = (1 + 1/apla) * W;
        W = project_simplex(T');
        W = (W + W') / 2;
        time_cost = toc;
        actual_ids = spectral_clustering(W, K);
        acc = accuracy(gnd', actual_ids);
        if(size(actual_ids, 2) == 1)
            actual_ids = actual_ids';
        end        
        cluster_data = cell(1, K);
        for pos_idx =  1 : K
             cluster_data(1, pos_idx) = { gnd(actual_ids(1, :) == pos_idx) };
        end
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
        disp([lambda, apla, acc, nmi, purity, fmeasure, ri, ari time_cost]);
        dlmwrite('allr_clustering_isolet.txt', [lambda, apla, acc, nmi, purity, fmeasure, ri, ari time_cost], '-append', 'delimiter', '\t', 'newline', 'pc');
    end    
end

