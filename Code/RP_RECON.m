function [O, x_recon,x_proj] = RP_RECON(X, k)
%RP_RECON This code implements the random projection (RP) method
%
% [O,x_recon,x_proj] = RP_RECON(X, k)
%
% INPUT
%   X       n x d matrix of length n and size d, where n refers to the 
%           number of observed timesteps and d the number of time series.
%   k       compression dimensionality, i.e. d time series are projected 
%           onto a k-dimensional base.
%
% OUTPUT
%   O       n x 1 vector of outlier scores. 
%   x_recon n x d matrix representing the model obtained by reconstruction
%           from the projection of all x in X. Note, x_recon is the
%           unscaled reconstruction.
%   x_proj  n x k matrix representing the lower-dimensional representation
%           of X.
%
% DESCRIPTION
% This code implements the random projection method to find outliers in 
% multivariate time series from reconstruction errors as retrieved by
% projecting the data onto k approximately orthonormal projection vectors.
%
%  Copyright: Madelon Hulsebos, m.hulsebos-1@student.tudelft.nl
%  Intelligent Systems Department, Pattern Recognition & Bioinformatics
%  Research Group.
%  Faculty of Electrical Engineering, Mathematics and Computer Science,         
%  Delft University of Technology,            
%  The Netherlands.
%

    % Number of data points n, number of time series d
    [n, d] = size(X);
    % Generation of k approximate orthonormal random vectors
    R = randn(d,k);
    W_r = 1/sqrt(d) .* R; 
    
    % For each arriving sample
    for i = 1 : n
        
        % The measurement vector at timestep i
        x_i = X(i,:)';
        % Project current measurement vector onto random base
        x_proj(:,i) = (W_r' * x_i); 
        % Reconstruct the vector to original dimensionality
        x_recon(:,i) = (W_r * x_proj(:,i)); % 
        % Compute the residual between original and reconstructed data
        % point
        x_residual = abs(x_t - x_recon(:,i));
        
        O(i,1) = norm(x_residual)^2;

    end
end
