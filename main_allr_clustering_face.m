close all;
clear;
clc;

addpath('data');
addpath('tools');

data = load('ExtendedYaleB.mat');
X = data.EYALEB_DATA;
gnd = data.EYALEB_LABEL;
K = max(gnd);
n = size(X, 2);

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end
for idx = 1 : n
    X(:, idx) = X(:, idx) ./ max(1e-12,  norm(X(:, idx)));
end

lamdas = [2];
aplas = [0.05];
apla_len = length(aplas);
lmd_len = length(lamdas);

XtX = X' * X;
D = calculate_similarity(X');
for lmd_idx = 1 : lmd_len
    lambda = lamdas(lmd_idx);
    Z = zeros(n, n);
    for pos =  1 : n
        Z(:, pos) = (XtX + lambda * diag(D(:, pos))) \ XtX(:, pos);
    end
    for i = 1 : n
        Z(:, i) = Z(:, i) ./ max(1e-12, sum(Z(:, i)));
    end
    for apla_idx = 1 : apla_len
        apla = aplas(apla_idx);
    
        tic;
        W = Z;
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
            cluster_data(1, pos_idx) = { gnd(actual_ids == pos_idx) };
        end 
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
        disp([lambda, apla, acc, nmi, purity, fmeasure, ri, ari, time_cost]);
        dlmwrite('allr_clustering_face.txt', [lambda, apla, acc, nmi, purity, fmeasure, ri, ari, time_cost], '-append', 'delimiter', '\t', 'newline', 'pc'); 
    end
    
end


