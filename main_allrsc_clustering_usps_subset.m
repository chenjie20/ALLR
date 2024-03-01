close all;
clear;
clc;

addpath('data');
addpath('tools');

load('usps.mat');

%generate data
K = 10;
n = 1000;
X = mat2gray(data(1 :n, 2 : end))';
gnd = data(1 : n, 1)';

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

for idx = 1 : n
    X(:, idx) = X(:, idx) ./ max(1e-12,  norm(X(:, idx)));
end

lamdas = [25];
aplas = [100]; 
apla_len = length(aplas);
lmd_len = length(lamdas);

D = calculate_similarity(X');
XtX = X' * X;
for lmd_idx = 1 : lmd_len
    lambda = lamdas(lmd_idx);
    % An alternative initialization
    Z = (XtX + lambda * eye(n)) \ XtX;
    for i = 1 : n
        Z(:, i) = Z(:, i) ./ max(1e-12, sum(Z(:, i)));
    end
     tic;
    for apla_idx = 1 : apla_len
        apla = aplas(apla_idx);
        W = Z';
        last_result = 0;
        for idx = 1 : apla
            T = (1 + 1/idx) * W;
            [W, sigma] = project_simplex(T);
            W = (W + W') / 2;
%                 current_result =  norm(W - T);
%                 result = (last_result - current_result);
%                 last_result = current_result;
%                 num = sum(W(:) > 1e-6);
%                 ratio = num / (n * n);
%                 if idx > 1
%                     disp([apla_idx ratio result]);
%                 end
        end    
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
        dlmwrite('allrsc_clustering_usps_subset.txt', [lambda, apla, acc, nmi, purity, fmeasure, ri, ari time_cost], '-append', 'delimiter', '\t', 'newline', 'pc');
    end
end
