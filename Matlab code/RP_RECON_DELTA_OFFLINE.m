function Ofinal = RP_RECON_DELTA_OFFLINE(X, m)
%RP_RECON This code implements the deltaRP for the specific purpose of reconstruction-based outlier detection.
%
% Ofinal = RP_RECON_DELTA_OFFLINE(X, m)
%
% INPUT
%   X       n x d matrix of length n and size d, where n refers to the number of observed timesteps and d the number of time series.
%   m       number of predictors for each data point.
%
% OUTPUT
%   O       n x 1 vector of outlier scores. 
%
% DESCRIPTION
% This code implements the deltaRP method to find outliers in multivariate time series from running the RP method twice, with k=1 and
% k=2. The outlier scores corresponds to the difference between the two obtained outlier scores from RP(k=1) and RP(k=2).
%
%  Copyright: Madelon Hulsebos, m.hulsebos-1@student.tudelft.nl
%  Intelligent Systems Department, Pattern Recognition & Bioinformatics Research Group.
%  Faculty of Electrical Engineering, Mathematics and Computer Science, Delft University of Technology, The Netherlands.
    
    % Number of data points n, number of time series d
    [n, d] = size(X);

    % Generation of k approximate orthonormal random vectors
    for i = 1 : m
        R1(:,:,i) = randn(d,1);
        R2(:,:,i) = randn(d,2);
    end
    W1 = 1/sqrt(d) .* R1; 
    W2 = 1/sqrt(d) .* R2;     
    
    for i = 1 : n
        
        % The measurement vector at timestep i
        x_i = X(i,:)';
        
        for j = 1 : m

            % Project current measurement vector onto random base
            x_proj1 = (W1(:,:,j)' * x_i); 
            x_proj2 = (W2(:,:,j)' * x_i);
            % Reconstruct the vector to original dimensionality
            x_recon1 = (W1(:,:,j) * x_proj1); 
            x_recon2 = (W2(:,:,j) * x_proj2);
            % Compute the residual between original and reconstructed data
            % point
            x_residual1 = abs(x_i - x_recon1);
            x_residual2 = abs(x_i - x_recon2);

            O_sub1(i,j) = norm(x_residual1)^2;
            O_sub2(i,j) = norm(x_residual2)^2;
            
        end
       
    end
    
   O_sub = normc(abs(normc(O_sub1) - normc(O_sub2)));
            
   Ofinal = max(O_sub,[],2);
    
end
