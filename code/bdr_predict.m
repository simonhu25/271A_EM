% Written by Jun Hao Hu, University of California San Diego.
% All rights reserved.

%-----------------------------------------------------------------------
% MATLAB routine that performs prediction using BDR and calculates the
% errors as well.
%-----------------------------------------------------------------------

function [A] = bdr_predict(dct_coeffs,dim,mu_fg,mu_bg,sigma_fg,sigma_bg,pi_fg,pi_bg,p_fg,p_bg,rows,cols,n_class)
A = zeros(rows,cols);

for idx_x = 1:rows
    for idx_y = 1:cols
        idx = cols*(idx_x-1)+idx_y;
        dct_coeff = dct_coeffs(idx,:);
        A(idx_x,idx_y) = p_fg*calc_prob(dct_coeff,dim,n_class,mu_fg,sigma_fg,pi_fg) > ...
            p_bg*calc_prob(dct_coeff,dim,n_class,mu_bg,sigma_bg,pi_bg);
    end
end
end