function Ofinal = RP_RECON_DELTA(X, m)
%RP_RECON_DELTA This code implements the deltaRP method for the specific purpose of reconstruction-based outlier detection.
%
% Ofinal = RP_RECON_DELTA(X, m)
%
% INPUT
%   X       n x d matrix of length n and size d, where n refers to the number of observed timesteps and d the number of time series.
%   m       number of RP_RECON predictors for each data point.
%
% OUTPUT
%   Ofinal  n x 1 vector of outlier scores. 
%
% DESCRIPTION
% This code implements the deltaRP method to find outliers in multivariate time series from running the RP method twice, with k=1 and
% k=2. The outlier scores corresponds to the difference between the two obtained outlier scores from RP(k=1) and RP(k=2).
%
% Copyright: Madelon Hulsebos, m.hulsebos-1@student.tudelft.nl
%  Intelligent Systems Department, Pattern Recognition & Bioinformatics Research Group.
%  Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, The Netherlands.
    
    % Number of data points n, number of time series d
    [n, d] = size(X);

    % For each arriving sample
    m1 = zeros(1,m);
    m2 = zeros(1,m);
    S1 = zeros(1,m);
    S2 = zeros(1,m);
    O_m = zeros(1,m);
    O_S = zeros(1,m);
    
    %   Generation of k approximate orthonormal random vectors
    for i = 1 : m
        R1(:,:,i) = randn(d,1);
        R2(:,:,i) = randn(d,2);
    end
    W1 = 1/sqrt(d) .* R1; 
    W2 = 1/sqrt(d) .* R2; 
    
    for i = 1 : n
        
        % The measurement vector at timestep t
        x_i = X(i,:)';
        
        for j = 1 : m

            % Project current measurement vector onto random base
            x_proj1 = (W1(:,:,j)' * x_i); 
            x_proj2 = (W2(:,:,j)' * x_i);
            % Reconstruct the vector to original dimensionality
            x_recon1 = (W1(:,:,j) * x_proj1); 
            x_recon2 = (W2(:,:,j) * x_proj2);
            % Compute the residual between original and reconstructed data point
            x_residual1 = abs(x_i - x_recon1);
            x_residual2 = abs(x_i - x_recon2);

            O_sub1 = norm(x_residual1)^2;
            O_sub2 = norm(x_residual2)^2;
            

            if i == 1
                m1_prev(j) = m1(j);
                m2_prev(j) = m2(j);                 
                S1(j) = 0;
                S2(j) = 0;
                
                O_sub(i,j) = 0;
             else
                 
                m1(j) = m1(j) +  (O_sub1 - m1(j)) / i;
                m2(j) = m2(j) +  (O_sub2 - m2(j)) / i;
                S1(j) = S1(j) +  (O_sub1 - m1(j)) * (O_sub1 - m1_prev(j));
                S2(j) = S2(j) +  (O_sub2 - m2(j)) * (O_sub2 - m2_prev(j));
                Std1 = sqrt(S1(j)/i);
                Std2 = sqrt(S2(j)/i);
                
                m1_prev(j) = m1(j);
                m2_prev(j) = m2(j);

                O_sub(i,j) = abs(((O_sub1 - m1(j)) / Std1) - ((O_sub2 - m2(j)) / Std2));
             end
            
        end
        
        if i == 1
            
            O_mprev = O_m;
            O_norm(i,:) = O_sub(i,:);
            
        else    
            
            O_mprev = O_m;
            O_m = O_m + (O_sub(i,:) - O_m) ./ i;
            O_S = O_S + (O_sub(i,:) - O_m) .* (O_sub(i,:) - O_mprev);

            O_norm(i,:) = (O_sub(i,:) - O_m) ./ sqrt(O_S ./ i);
            
        end
        
        Ofinal(i) = max(O_norm(i,:));
    end
    
end
