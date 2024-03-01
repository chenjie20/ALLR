close all;
clear;
clc;

addpath('data');
addpath('tools');

data = load('ExtendedYaleB.mat');
sample_data = data.EYALEB_DATA;
sample_lables = data.EYALEB_LABEL;
K = max(sample_lables); 
n = size(sample_data, 2);
rum_times = 10;
ratios = [0.05, 0.1, 0.2, 0.5];

lamdas = [2e6];
aplas = [3];

% a parameter for the third-part classification algortihm
mus = [0.99];

apla_len = length(aplas);
lmd_len = length(lamdas);
mu_len = length(mus);
ratio_len = length(ratios);

X = sample_data;

DD = calculate_similarity(X');
XtX = X' * X;
for lmd_idx = 1 : lmd_len
    lambda = lamdas(lmd_idx);
    Z = zeros(n, n);    
    for pos =  1 : n
        Z(:, pos) = (XtX + lambda * diag(DD(:, pos))) \ (X' * X(:, pos));
    end
    for i = 1 : n
        Z(:, i) = Z(:, i) ./ max(1e-12, sum(Z(:, i)));
    end
    for apla_idx = 1 : apla_len
        apla = aplas(apla_idx);        
        W = Z';
        last_result = 0;
        for idx = 1 : apla
            T = (1 + 1/idx) * W;
            [W, sigma] = project_simplex(T);
            W = (W + W') / 2;
        end
        
        %the third-part classification algortihm
        for ratio_idx = 1 : ratio_len
            ratio = ratios(ratio_idx);
            for mu_idx = 1 : mu_len
                mu = mus(mu_idx);
                results = zeros(1, rum_times);       
                for it = 1 : rum_times
                        total_num_class = floor(n / K); % number of data per class
                        label_num_class = floor(ratio * total_num_class); % number of labeled data per class
                        label_index_class = sort(randperm(total_num_class, label_num_class)); % index of labeled data selected
                        label_index = zeros(1, label_num_class * K); % labelind: index of known label
                        for i = 1 : K
                            st = (i - 1) * label_num_class + 1;
                            ed = i * label_num_class;
                            label_index(st : ed) = label_index_class + (i - 1) * total_num_class;
                        end
                        unknown_label_index = setdiff(1 : n, label_index);%index of unlabeled data
                        %sync notation
                        mm = length(label_index);
                        labels = zeros(mm, K);
                        for i = 1:mm
                            labels(i, sample_lables(label_index(i))) = 1;
                        end
                        known_nodes = label_index;
                        nodes_to_predict = unknown_label_index;
                        options.alpha = mu;
                        %label
                        if ~isfield(options, 'precision')
                            options.precision = 1e-5;
                        end
                        if ~isfield(options, 'maxiter')
                            options.maxiter = 100;
                        end

                        % Normalization of the adjacency matrix
                        D = diag(sum(W, 2).^(-0.5));
                        diffusion_matrix = D * W * D;

                        k = size(labels, 2);
                        ini_scores = zeros(n, k);
                        ini_scores(known_nodes, :) = labels;
                        predictions_score = ini_scores;

                        % Propagation
                        for i=1:options.maxiter
                            last_score = predictions_score;
                            predictions_score = options.alpha*diffusion_matrix*predictions_score + (1-options.alpha)*ini_scores;
                            if max(max(abs(last_score-predictions_score))) < options.precision
                                break;
                            end
                        end
                        % keep only predictions for nodes specified by 'nodes_to_predict'
                        predictions_score = predictions_score(nodes_to_predict, :);

                        % calculate accuracy
                        [ur, uc]=size(unknown_label_index);
                        [max_value, max_ind] = max(predictions_score,[],2);
                        cnt = 0;
                        for i = 1:uc
                            if max_ind(i) == sample_lables(unknown_label_index(i))
                                cnt = cnt+1;
                            end
                        end
                        results(it) = cnt/uc;       
                 end
                avg_acc = mean(results);
                std_acc = std(results);
                disp([ratio lambda / 1e4 apla avg_acc std_acc]);
                dlmwrite('classification_allrsc_yaleb.txt', [ratio lambda apla options.alpha avg_acc std_acc], '-append', 'delimiter', '\t', 'newline', 'pc');
            end
        end
    end
end