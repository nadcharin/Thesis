
%% Generate plots of signal
t = 1:0.05:10;
n = size(t,2);
d = 50;
s_norm = [];

for i = 1:d
    si = 2 .* rand(1) + sin(pi * t);
    si = si + (0.05 .* randn(1,n));
    s_norm = [s_norm; si];
end

for i = 1:d
    si = 2 .* rand(1) + cos(pi * t);
    si = si + (0.05 .* randn(1,n));
    s_norm = [s_norm; si];
end

for i = 1:d
    si = 2 .* rand(1) + cos(pi * t) .* sin(pi * t);
    si = si + (0.05 .* randn(1,n));
    s_norm = [s_norm; si];
end

lab = zeros(1,size(s_norm,2));
%s = s_norm;

% Create independent test data

s = [];

for i = 1:d
    si = 2 .* rand(1) + sin(pi * t);
    si = si + (0.05 .* randn(1,n));
    s = [s; si]; % 
end

for i = 1:d
    si = 2 .* rand(1) + cos(pi * t);
    si = si + (0.05 .* randn(1,n));
    s = [s; si]; % 
end

for i = 1:d
    si = 2 .* rand(1) + cos(pi * t) .* sin(pi * t);
    si = si + (0.05 .* randn(1,n));
    s = [s; si]; % 
end

lab = zeros(1,size(s,2));

% s_norm = s_norm - mean(s_norm,2);
% s = s - mean(s,2);
% figure; plot(s_norm(1,:)); hold on;
%plot(s(1,:));

%%


time_dim_rp = [];
time_dim_pca = [];

for l = 1:6
    
    auc_runs = [];
    auc_detectors = [];  
    
    detected_point = [];
    detected_sequential = [];
    
    auc_runs_pca = [];
    auc_detectors_pca  = [];  
    
    detected_point_pca  = [];
    detected_sequential_pca  = [];
    
    time_rp = [];
    time_pca = [];

    for r = 1:100

        % Reset outlier-injected test data to independent uninjected test data
        s_run = s;

        %% Retrieving subsets of signals, which will be injected with outliers
        m = size(s_run,1);

        sub_sig1 = randperm(m, m/5);
        sub_sig2 = randperm(m, m/5);
        sub_sig3 = randperm(m, m/5);
        sub_sig4 = randperm(m, m/5);
        sub_sig5 = randperm(m, m/5);
        sub_sig6 = randperm(m, m/5);

        %% Introducing sequential outliers
        fault_val = s_run(sub_sig1, 50);
        s_run(sub_sig1, 50:65) = repmat(fault_val,1, 16);
        fault_val2 = s_run(sub_sig3, 50);
        s_run(sub_sig3, 50:65) = repmat(fault_val,1, 16);

        fault_val = s_run(sub_sig2, 140);
        s_run(sub_sig2, 140:155) = repmat(fault_val,1, 16);
        fault_val1 = s_run(sub_sig5, 140);
        s_run(sub_sig5, 140:155) = repmat(fault_val,1, 16);

        % figure; plot(s(sub_sig1,:)', 'LineWidth', 1);

        lab(1, 50:65) = 1;
        lab(1, 140:155) = 1;

        %% Introducing point outliers
        fault_val = 1.3 * s_run(sub_sig4,[70:72, 80:82, 90:92, 116:118]);
        s_run(sub_sig4, [70:72, 80:82, 90:92, 116:118]) = fault_val;

        lab(1, [70:72, 80:82, 90:92, 116:118]) = 1;

        fault_val = 1.3 * s_run(sub_sig6, [11:13, 31:33, 163:165, 176:178]);
        s_run(sub_sig6, [11:13, 31:33, 163:165, 176:178]) = fault_val;

        lab(1, [11:13, 31:33, 163:165, 176:178]) = 1;
        
        % figure; plot(s(sub_sig4,:)', 'LineWidth', 1);

        %% Resulting data visualisation

        % Scatters of t against t+1
        % s_t = s(:,1:end-1);
        % s_t_plus1 = s(:,2:end);

        % Sequential outlier
        % figure; plot(t, s(sub_sig1,:));
        % figure; scatter(s_t(sub_sig1(1),:), s_t_plus1(sub_sig1(1),:));

        % Point outlier
        % figure; plot(t, s(sub_sig4(1),:));
        % figure; scatter(s_t(sub_sig4(1),:), s_t_plus1(sub_sig4(1),:));

        %% Clustering the 'sensor signals' based on their 'fault sensitivity' on 
        % the injected outliers: 

        rand_idx = randperm(m);

        ens1 = rand_idx(1:1*m/5);
        ens2 = rand_idx(1*m/5+1:2*m/5);
        ens3 = rand_idx(2*m/5+1:3*m/5);
        ens4 = rand_idx(3*m/5+1:4*m/5);
        ens5 = rand_idx(4*m/5+1:5*m/5);

        sub_ens1_norm = s_norm(ens1,:);
        sub_ens2_norm = s_norm(ens2,:);
        sub_ens3_norm = s_norm(ens3,:);
        sub_ens4_norm = s_norm(ens4,:);
        sub_ens5_norm = s_norm(ens5,:);

        sub_ens1 = s_run(ens1,:);
        sub_ens2 = s_run(ens2,:);
        sub_ens3 = s_run(ens3,:);
        sub_ens4 = s_run(ens4,:);
        sub_ens5 = s_run(ens5,:);

        m_sub = size(ens1,2);
        k = l;
        
        tic;
        
        R1 = randn(m_sub, k);
        R2 = randn(m_sub, k);
        R3 = randn(m_sub, k);
        R4 = randn(m_sub, k);
        R5 = randn(m_sub, k);

        s_norm1_comp = (1/k^(1/2)) * sub_ens1_norm' * R1;
        s_norm2_comp = (1/k^(1/2)) * sub_ens2_norm' * R2;
        s_norm3_comp = (1/k^(1/2)) * sub_ens3_norm' * R3;
        s_norm4_comp = (1/k^(1/2)) * sub_ens4_norm' * R4;
        s_norm5_comp = (1/k^(1/2)) * sub_ens5_norm' * R5;

        s1_comp = (1/k^(1/2)) * sub_ens1' * R1;
        s2_comp = (1/k^(1/2)) * sub_ens2' * R2;
        s3_comp = (1/k^(1/2)) * sub_ens3' * R3;
        s4_comp = (1/k^(1/2)) * sub_ens4' * R4;
        s5_comp = (1/k^(1/2)) * sub_ens5' * R5;
        
        
        % Plot the behavior
    %     figure; plot([s_norm1_comp, s_norm2_comp, s_norm3_comp, s_norm4_comp, s_norm5_comp]); 
    %     
    %     % Scatters of t against t+1 of compressed signals first detector
    %     s_t = s1_comp(1:end-1,:);
    %     s_t_plus1 = s1_comp(2:end,:);
    %     
    %     % Sequential outlier
    %     figure; plot(s1_comp');
    %     figure; scatter(s_t, s_t_plus1, [], lab(1:end-1)');
    % 
    %     s_t = s2_comp(1:end-1,:);
    %     s_t_plus1 = s2_comp(2:end,:);
    %     
    %     % Sequential outlier
    %     figure; plot(s2_comp');
    %     figure; scatter(s_t, s_t_plus1, [], lab(1:end-1)');
        %%
        s_norm1_comp = s_norm1_comp - mean(s_norm1_comp);
        s_norm2_comp = s_norm2_comp - mean(s_norm2_comp);
        s_norm3_comp = s_norm3_comp - mean(s_norm3_comp);
        s_norm4_comp = s_norm4_comp - mean(s_norm4_comp);
        s_norm5_comp = s_norm5_comp - mean(s_norm5_comp);

        s1_comp = s1_comp - mean(s1_comp);
        s2_comp = s2_comp - mean(s2_comp);
        s3_comp = s3_comp - mean(s3_comp);
        s4_comp = s4_comp - mean(s4_comp);
        s5_comp = s5_comp - mean(s5_comp);

    %     figure; plot(s_norm1_comp); hold on;
    %     plot(s1_comp);
    %     figure; plot(s_norm2_comp); hold on;
    %     plot(s2_comp);
    %     figure; plot(s_norm3_comp); hold on;
    %     plot(s3_comp);
    %     figure; plot(s_norm4_comp); hold on;
    %     plot(s4_comp);
    %     figure; plot(s_norm5_comp); hold on;
    %     plot(s5_comp);

        %% Residual-based detection

        % Compute the residual of the learned compressed model and the compressed 
        % measurements.
        residual1 = abs(s_norm1_comp - s1_comp);
        residual2 = abs(s_norm2_comp - s2_comp);
        residual3 = abs(s_norm3_comp - s3_comp);
        residual4 = abs(s_norm4_comp - s4_comp);
        residual5 = abs(s_norm5_comp - s5_comp);

    %     figure; plot([residual1, residual2, residual3, residual4, residual5]);
    
        % Retrieve all otimal thresholds acc. to ROC curve
        thres1 = mean(residual1) + std(residual1);
        thres2 = mean(residual2) + std(residual2);
        thres3 = mean(residual3) + std(residual3);
        thres4 = mean(residual4) + std(residual4);
        thres5 = mean(residual5) + std(residual5);

        % Perform detections: we only want to communicate in case something is
        % detected (to save communication overhead. Therefore the detections are
        % done in the early stage.
        
        detection1 = +(residual1 >= thres1);
        detection1_sum = sum(detection1 - (thres1),2);
        detection1_mean = mean(detection1_sum);
        detection1_std = std(detection1_sum);
        detection1 = zeros(181,l);
        detection1(detection1_sum >= detection1_mean + detection1_std) = +(sum(residual1((detection1_sum >= detection1_mean + detection1_std),:),2));
        detection1(detection1_sum < detection1_mean + detection1_std) = 0;
        
        detection2 = +(residual2 >= thres2);
        detection2_sum = sum(detection2 - (thres2),2);
        detection2_mean = mean(detection2_sum);
        detection2_std = std(detection2_sum);
        detection2 = zeros(181,l);
        detection2(detection2_sum >= detection2_mean + detection2_std) = +(sum(residual2((detection2_sum >= detection2_mean + detection2_std),:),2));% - (lab2_mean + lab2_std));
        detection2(detection2_sum < detection2_mean + detection2_std) = 0;

        detection3 = +(residual3 >= thres3);
        detection3_sum = sum(detection3 - (thres3),2);
        detection3_mean = mean(detection3_sum);
        detection3_std = std(detection3_sum);
        detection3 = zeros(181,l);
        detection3(detection3_sum >= detection3_mean + detection3_std) = +(sum(residual3((detection3_sum >= detection3_mean + detection3_std),:),2));% - (lab3_mean + lab3_std));
        detection3(detection3_sum < detection3_mean + detection3_std) = 0;

        detection4 = +(residual4 >= thres4);
        detection4_sum = sum(detection4 - (thres4),2);
        detection4_mean = mean(detection4_sum);
        detection4_std = std(detection4_sum);
        detection4 = zeros(181,l);
        detection4(detection4_sum >= detection4_mean + detection4_std) = +(sum(residual4((detection4_sum >= detection4_mean + detection4_std),:),2));% - (lab4_mean + lab4_std));
        detection4(detection4_sum < detection4_mean + detection4_std) = 0;

        detection5 = +(residual5 >= thres5);
        detection5_sum = sum(detection5 - (thres5),2);
        detection5_mean = mean(detection5_sum);
        detection5_std = std(detection5_sum);
        detection5 = zeros(181,l);
        detection5(detection5_sum >= detection5_mean + detection5_std) = +(sum(residual5((detection5_sum >= detection5_mean + detection5_std),:),2));% - (lab5_mean + lab5_std));
        detection5(detection5_sum < detection5_mean + detection5_std) = 0;
        
        
%         [pos, ~] = find(residual1 >= thres1);
%         [neg, ~] = setdiff(1:181,unique(pos));
%         sum1 = sum(abs(residual1(unique(pos),:) - thres1),2);
%         mean1 = mean(sum1);
%         std1 = std(sum1);
%         detection1(sum1 >= mean1+std1,1) = +sum(residual1(sum1 >= mean1 + std1,:),2);
%         detection1(setdiff(1:181,sum1 >= mean1+std1),1) = 0;
%         
%         [pos, ~] = find(residual2 >= thres2);
%         [neg, ~] = setdiff(1:181,unique(pos));
%         sum2 = sum(abs(residual2(unique(pos),:) - thres2),2);
%         mean2 = mean(sum2);
%         std2 = std(sum2);
%         detection2(sum2 >= mean2+std2,1) = +sum(residual2(sum2 >= mean2 + std2,:),2);
%         detection2(setdiff(1:181,sum2 >= mean2+std2),1) = 0;
%         
%         [pos, ~] = find(residual3 >= thres3);
%         [neg, ~] = setdiff(1:181,unique(pos));
%         sum3 = sum(abs(residual3(unique(pos),:) - thres3),2);
%         mean3 = mean(sum3);
%         std3 = std(sum3);
%         detection3(sum3 >= mean3+std3,1) = +sum(residual3(sum3 >= mean3 + std3,:),2);
%         detection3(setdiff(1:181,sum3 >= mean3+std3),1) = 0;
%         
%         [pos, ~] = find(residual4 >= thres4);
%         [neg, ~] = setdiff(1:181,unique(pos));
%         sum4 = sum(abs(residual4(unique(pos),:) - thres4),2);
%         mean4 = mean(sum4);
%         std4 = std(sum4);
%         detection4(sum4 >= mean4+std4,1) = +sum(residual4(sum4 >= mean4 + std4,:),2);
%         detection4(setdiff(1:181,sum4 >= mean4+std4),1) = 0;
%         
%         [pos, ~] = find(residual5 >= thres5);
%         [neg, ~] = setdiff(1:181,unique(pos));
%         sum5 = sum(residual5(unique(pos),:) - thres5,2);
%         mean5 = mean(sum5);
%         std5 = std(sum5);
%         detection5(sum5 >= mean5+std5,1) = +sum(residual5(sum5 >= mean5 + std5,:),2);
%         detection5(setdiff(1:181,sum5 >= mean5+std5),1) = 0;
        
        % End of continuous operations; from here only needed in case
        % samples are flagged as outliers.

        % Note that only samples flagged with values > 0 are communicated to the 
        % final detector. Therefore communication costs are being reduced. As the
        % runtime is not being measured at this point, the detections are just
        % added.
        final_detection = max([detection1 detection2 detection3 detection4 detection5],[],2);
        %final_detection(final_detection~=0) =  1;
        % Also note, that the outlier scores are not normalized, and stem directly
        % from the residual value.

        %%% TO DO: BUILD IN ENSEMBLE DECISION %%%

        [x, y, thres, auc, optrocpt] = perfcurve(lab, final_detection, 1, 'NegClass', 0, 'XCrit', 'FPR', 'YCrit', 'TPR');
        auc_runs = [auc_runs; auc];
        
        thres_final = thres(find(x(:,1) == optrocpt(1,1) & y(:,1) == optrocpt(1,2)));
        
        final_detection = (final_detection >= thres_final);
       
    %     Individual learner ROCs
    %     figure; plot([x1,x2,x3,x4,x5],[y1,y2,y3,y4,y5], 'LineWidth', 1); hold on;
    %     Ensemble learner ROC
    %     figure; plot(x, y, 'LineWidth', 1); hold on;

    %     
    %     detection = zeros(1,size(final_detection,2));
    %     thres_opt = thres(x(:,1) == optrocpt(1,1) & y(:,1) == optrocpt(1,2));
    %     detection(final_detection >= thres_opt) = 1;
    %     figure; plot(detection); hold on;
    %     plot(lab);

        b = toc;
        time_rp = [time_rp; b];
        
        detected_point = [detected_point; sum(final_detection([11:13, 31:33, 163:165, 176:178, 70:72, 80:82, 90:92, 116:118],1))];
        detected_sequential = [detected_sequential; sum(final_detection([50:65, 140:155],1))];
        
        tic; 
        
        % Compute mapping from normal behavior
        [pca_norm1, score1, ~, ~, ~, mu1] = pca(sub_ens1_norm', 'numcomponents', l);
        [pca_norm2, score2, ~, ~, ~, mu2] = pca(sub_ens2_norm', 'numcomponents', l);
        [pca_norm3, score3, ~, ~, ~, mu3] = pca(sub_ens3_norm', 'numcomponents', l);
        [pca_norm4, score4, ~, ~, ~, mu4] = pca(sub_ens4_norm', 'numcomponents', l);
        [pca_norm5, score5, ~, ~, ~, mu5] = pca(sub_ens5_norm', 'numcomponents', l);

        % Mapped data to the one component.
        s1_pca =  sub_ens1' * pca_norm1;
        % Reconstruct 
        s1_comp = s1_pca * pca_norm1' + repmat(mu1, size(sub_ens1_norm, 2), 1);

        % Mapped data to the one component.
        s2_pca =  sub_ens2' * pca_norm2;
        % Reconstruct 
        s2_comp = s2_pca * pca_norm2' + repmat(mu2, size(sub_ens2_norm, 2), 1);

        % Mapped data to the one component.
        s3_pca =  sub_ens3' * pca_norm3;
        % Reconstruct 
        s3_comp = s3_pca * pca_norm3' + repmat(mu3, size(sub_ens3_norm, 2), 1);

        % Mapped data to the one component.
        s4_pca =  sub_ens4' * pca_norm4;
        % Reconstruct 
        s4_comp = s4_pca * pca_norm4' + repmat(mu4, size(sub_ens4_norm, 2), 1);

        % Mapped data to the one component.
        s5_pca =  sub_ens5' * pca_norm5;
        % Reconstruct 
        s5_comp = s5_pca * pca_norm5' + repmat(mu5, size(sub_ens5_norm, 2), 1);

        % Plot the behavior
    %     figure; plot([s_norm1_comp, s_norm2_comp, s_norm3_comp, s_norm4_comp, s_norm5_comp]); 
    %     
    %     % Scatters of t against t+1 of compressed signals first detector
    %     s_t = s1_comp(1:end-1,:);
    %     s_t_plus1 = s1_comp(2:end,:);
    %     
    %     % Sequential outlier
    %     figure; plot(s1_comp');
    %     figure; scatter(s_t, s_t_plus1, [], lab(1:end-1)');
    % 
    %     s_t = s2_comp(1:end-1,:);
    %     s_t_plus1 = s2_comp(2:end,:);
    %     
    %     % Sequential outlier
    %     figure; plot(s2_comp');
    %     figure; scatter(s_t, s_t_plus1, [], lab(1:end-1)');

    %     figure; plot(s_norm1_comp); hold on;
    %     plot(s1_comp);
    %     figure; plot(s_norm2_comp); hold on;
    %     plot(s2_comp);
    %     figure; plot(s_norm3_comp); hold on;
    %     plot(s3_comp);
    %     figure; plot(s_norm4_comp); hold on;
    %     plot(s4_comp);
    %     figure; plot(s_norm5_comp); hold on;
    %     plot(s5_comp);

        %% Residual-based detection

        % Compute the residual of the compressed model and the compressed 
        % measurements.
        residual1 = abs(s1_comp' - sub_ens1);
        mean1 = mean(residual1,2);
        std1 = std(residual1,0,2);
        residual2 = abs(s2_comp' - sub_ens2);
        mean2 = mean(residual2,2);
        std2 = std(residual2,0,2);
        residual3 = abs(s3_comp' - sub_ens3);
        mean3 = mean(residual3,2);
        std3 = std(residual3,0,2);
        residual4 = abs(s4_comp' - sub_ens4);
        mean4 = mean(residual4,2);
        std4 = std(residual4,0,2);
        residual5 = abs(s5_comp' - sub_ens5);
        mean5 = mean(residual5,2);
        std5 = std(residual5,0,2);

        % Now still each 'sensor' provides a score on the data point. No 
        % compression has taken place!
        lab1 = +(residual1 >= mean1 + std1);
        lab1_sum = sum(lab1 - (mean1 + std1));
        lab1_mean = mean(lab1_sum);
        lab1_std = std(lab1_sum);
        lab1 = [];
        lab1(lab1_sum >= lab1_mean + lab1_std) = +(sum(residual1(:,(lab1_sum >= lab1_mean + lab1_std)),1));
        lab1(lab1_sum < lab1_mean + lab1_std) = 0;

        lab2 = +(residual2 >= mean2 + std2);
        lab2_sum = sum(lab2 - (mean2 + std2));
        lab2_mean = mean(lab2_sum);
        lab2_std = std(lab2_sum);
        lab2 = [];
        lab2(lab2_sum >= lab2_mean + lab2_std) = +(sum(residual2(:,(lab2_sum >= lab2_mean + lab2_std)),1));% - (lab2_mean + lab2_std));
        lab2(lab2_sum < lab2_mean + lab2_std) = 0;

        lab3 = +(residual3 >= mean3 + std3);
        lab3_sum = sum(lab3 - (mean3 + std3));
        lab3_mean = mean(lab3_sum);
        lab3_std = std(lab3_sum);
        lab3 = [];
        lab3(lab3_sum >= lab3_mean + lab3_std) = +(sum(residual3(:,(lab3_sum >= lab3_mean + lab3_std)),1));% - (lab3_mean + lab3_std));
        lab3(lab3_sum < lab3_mean + lab3_std) = 0;

        lab4 = +(residual4 >= mean4 + std4);
        lab4_sum = sum(lab4 - (mean4 + std4));
        lab4_mean = mean(lab4_sum);
        lab4_std = std(lab4_sum);
        lab4 = [];
        lab4(lab4_sum >= lab4_mean + lab4_std) = +(sum(residual4(:,(lab4_sum >= lab4_mean + lab4_std)),1));% - (lab4_mean + lab4_std));
        lab4(lab4_sum < lab4_mean + lab4_std) = 0;

        lab5 = +(residual5 >= mean5 + std5);
        lab5_sum = sum(lab5 - (mean5 + std5));
        lab5_mean = mean(lab5_sum);
        lab5_std = std(lab5_sum);
        lab5 = [];
        lab5(lab5_sum >= lab5_mean + lab5_std) = +(sum(residual5(:,(lab5_sum >= lab5_mean + lab5_std)),1));% - (lab5_mean + lab5_std));
        lab5(lab5_sum < lab5_mean + lab5_std) = 0;

        % End of continuous operations; from here only needed in case
        % samples are flagged as outliers.

        % Note that only samples flagged with values > 0 are communicated to the 
        % final detector. Therefore communication costs are being reduced. As the
        % runtime is not being measured at this point, the detections are just
        % added.
        final_detection = max([lab1; lab2; lab3; lab4; lab5],[],1);
        %final_detection(final_detection~=0) =  1;
        % Also note, that the outlier scores are not normalized, and stem directly
        % from the residual value.

        %%% TO DO: BUILD IN ENSEMBLE DECISION %%%

        [x, y, thres, auc, optrocpt] = perfcurve(lab, final_detection, 1, 'NegClass', 0, 'XCrit', 'FPR', 'YCrit', 'TPR');
        auc_runs_pca = [auc_runs_pca; auc];
        
        thres_final = thres(x(:,1) == optrocpt(1,1) & y(:,1) == optrocpt(1,2));
        
        final_detection = (final_detection >= thres_final);
        
        a=toc;
        time_pca = [time_pca; a];
        
        detected_point_pca = [detected_point_pca; sum(final_detection(1,[11:13, 31:33, 163:165, 176:178, 70:72, 80:82, 90:92, 116:118]))];
        detected_sequential_pca = [detected_sequential_pca; sum(final_detection(1,[50:65, 140:155],1))];
        
        
    end

    auc_runs_dim(:,l) = mean(auc_runs);
    std_runs_dim(:,l) = std(auc_runs);
    
    detected_point_dim(:,l) = mean(detected_point);
    detected_sequential_dim(:,l) = mean(detected_sequential);    
    
    auc_runs_dim_pca(:,l) = mean(auc_runs_pca);
    std_runs_dim_pca(:,l) = std(auc_runs_pca);
    
    detected_point_dim_pca(:,l) = mean(detected_point_pca);
    detected_sequential_dim_pca(:,l) = mean(detected_sequential_pca);
    
    time_dim_rp(l) = mean(time_rp);
    time_dim_pca(l) = mean(time_pca);
    
    std_dim_rp(l) = std(time_rp);
    std_dim_pca(l) = std(time_pca);
    
end

% lab_num_point = sum(lab(1,[11:13, 31:33, 163:165, 176:178, 70:72, 80:82, 90:92, 116:118]),2);
% lab_num_sequential = sum(lab(1,[50:65, 140:155]),2);
% perc_point = detected_point_dim / lab_num_point;
% perc_sequential = detected_sequential_dim / lab_num_sequential;
% 
% figure; plot(perc_point); hold on;
% plot(perc_sequential); hold off;

figure(1); errorbar(1:6,auc_runs_dim, std_runs_dim); hold on; 
figure(1); errorbar(1:6,auc_runs_dim_pca, std_runs_dim_pca); hold on; 

figure(2); errorbar(1:6,time_dim_rp, std_dim_rp); hold on;
figure(2); errorbar(1:6,time_dim_pca, std_dim_pca); hold on;
