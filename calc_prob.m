% Written by Jun Hao Hu, University of California San Diego.
% All rights reserved.

%-----------------------------------------------------------------------
% MATLAB routine that takes the data and calculates the posterior
% probability.
%-----------------------------------------------------------------------

function [dct_likelihood] = calc_prob(dct_coeffs,n_dim,n_class,mu,sigma,pi)
mu_c = zeros(n_class,n_dim);
sigma_c = zeros(n_class*n_dim,n_class*n_dim);
dct_coeffs = dct_coeffs(1:n_dim); % Only keep the first dim dimensions of the DCT coefficients.
dct_likelihood = 0;

for idx = 1:n_class
   split_1 = (idx-1)*64;
   split_2 = (idx-1)*64;
   mu_c(idx,:) = mu(split_1+1:split_2+n_dim);
   sigma_c((idx-1)*n_dim+1:idx*n_dim,(idx-1)*n_dim+1:idx*n_dim) = diag(sigma(split_1+1:split_2+n_dim)); % Diagonal covariance matrix.
   dct_likelihood = dct_likelihood + mvnpdf(...
       dct_coeffs,mu_c(idx,:),sigma_c((idx-1)*n_dim+1:idx*n_dim,(idx-1)*n_dim+1:idx*n_dim))...
       *pi(idx);
end
end