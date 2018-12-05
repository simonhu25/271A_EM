% Written by Jun Hao Hu, University of California San Diego. 
% All rights reserved. 

%-------------------------------------------------------------------
% MATLAB program to perform expectation-maximization (EM). 
% Input:
%   dct_coeffs : vector containing the DCT coefficients
%   n_class : int containing the number of classes
%   n_dim : int containing the number of dimensions we want
%   num_iter : int containing the maximum number of iterations for EM
% Output: 
%   mu : the EM estimate of the mean, mu
%   sigma : the EM estimate of the covariance, sigma
%   pi_c : the EM estimate of the weights for the mixture model
%-------------------------------------------------------------------

function [mu,sigma,pi_c] = em_calc(dct_coeffs,n_class,n_dim,num_iter)

% Assign parameters for EM algorithm.

[n_rows, ~] = size(dct_coeffs);
dct_coeffs_dim = dct_coeffs(:,1:n_dim);

sigma_c = diag(diag(2*rand(n_dim*n_class,n_dim*n_class)+2)); % Initialization of sigma.
mu_c = 3*rand(n_class,n_dim)+3; % Initialization of mu.
pi_c = (randi(20,1,n_class)); % Initialization of the weights, pi_c.
pi_c = pi_c/sum(pi_c); % Constraint equation.
epsilon = (1e-06)*ones(size(sigma_c)); % Small paramater to prevent zero covariance.
z = zeros(n_rows,n_class); % Initialization of the vector z.

% Start the EM algorithm.
for iter = 1:num_iter
    % Expectation step.
    z_old = z;
    for idx_exp = 1:n_rows
        p_x = zeros(1,n_class); % Pre-allocating space for likelihood of x.
        for idx_c = 1:n_class
            split_1 = (idx_c-1)*n_dim; % Splitting index 1 for convenience.
            split_2 = idx_c*n_dim; % Splitting index 2 for convenience. 
            sigma = sigma_c(split_1+1:split_2,split_1+1:split_2); % Update for sigma.
            mu = mu_c(idx_c,:); % Update for mu.
            p_x(idx_c) = mvnpdf(dct_coeffs_dim(idx_exp,:),mu,sigma)*...
                pi_c(idx_c); % Calculate the likelihood of x.
        end
       z(idx_exp,:) = p_x/sum(p_x); % Normalization of the vector.
    end
    pi_c = sum(z)/n_rows; % Update for pi_c.
    
    % Maximization step.
    for idx_c = 1:n_class
        split_1 = (idx_c-1)*n_dim; % Splitting index 1 for convenience.
        split_2 = idx_c*n_dim; % Splitting index 2 for convenience. 
        sig = (dct_coeffs_dim-repmat(mu_c(idx_c,:),n_rows,1));
        sigma = sig.*(repmat(z(:,idx_c),1,n_dim));
        tot = sum(z(:,idx_c));
        sigma_c(split_1+1:split_2,split_1+1:split_2) = (sigma'*sig)/tot;
        mu_c(idx_c,:) = sum(dct_coeffs_dim.*repmat(z(:,idx_c),1,n_dim))/tot;
    end
    sigma_c = diag(diag(sigma_c + epsilon));
    if (log(z) - log(z_old)) < 1e-06
           break;
    end
end

% Grab the correct parameters after EM calculation.
mu = zeros(1,n_dim*n_class);
for idx_c = 1:n_class
    mu((idx_c-1)*n_dim+1:idx_c*n_dim) = mu_c(idx_c,:);
end
sigma = diag(sigma_c).';
end