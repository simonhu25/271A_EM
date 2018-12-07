% Written by Jun Hao Hu, University of California San Diego.
% All rights reserved.

%---------------------------------------------------------------------
% Main MATLAB routine for homework 4, part 2.
%---------------------------------------------------------------------
%% Load data and define parameters. 
load('load_data.mat'); % MATLAB data array that contains information already loaded in the first section of em_main.

dims = [1 2 4 8 16 24 32 40 48 56 64]; % Desired dimensions.
[~,n_dims] = size(dims);
classes = [1 2 4 8 16 32];
n_classes = size(classes,2);
p_error = zeros(n_classes,n_dims);
fprintf('Loading completed. Training and predicting models ...\n');
%% EM model training and prediction.
for idx_class = 1:n_classes
   A_mask = zeros(rows_cheetah_ori,cols_cheetah_ori);
   fprintf('---------Starting class %f---------',idx_class);
   [mu_fg_c,sigma_fg_c,pi_fg_c] = em_calc(TrainsampleDCT_FG,classes(idx_class),M,200);
   [mu_bg_c,sigma_bg_c,pi_bg_c] = em_calc(TrainsampleDCT_BG,classes(idx_class),M,200);
   for idx_dim = 1:n_dims
       idx_dim
       A_mask = bdr_predict(dct_coeffs,dims(idx_dim),mu_fg_c,mu_bg_c,sigma_fg_c,sigma_bg_c,...
           pi_fg_c,pi_bg_c,p_fg,p_bg,rows_cheetah_ori,cols_cheetah_ori,classes(idx_class));
       p_error(idx_class,idx_dim) = calc_error(A_mask,p_fg,p_bg);
   end
end
