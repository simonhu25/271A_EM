% Written by Jun Hao Hu, University of California San Diego.
% All rights reserved.

%---------------------------------------------------------------------
% Main MATLAB routine for homework 4.
%---------------------------------------------------------------------

%% Load data and define parameters.
load('TrainingSamplesDCT_8_new.mat');

cheetah_ori = im2double(imread('cheetah.bmp')); % Convert the image into the range [0,1].
cheetah_pad = padarray(cheetah_ori, [4 4], 'both');
zig = load('Zig-Zag Pattern.txt');
zig = zig+1;
M = 64;

[rows_cheetah_ori,cols_cheetah_ori] = size(cheetah_ori);
[rows_cheetah_pad,cols_cheetah_pad] = size(cheetah_pad);
[rows_fg,~] = size(TrainsampleDCT_FG);
[rows_bg,~] = size(TrainsampleDCT_BG);

p_fg = rows_fg/(rows_fg+rows_bg); % Calculate the priors using the MLE.
p_bg = rows_bg/(rows_bg+rows_fg); % Calculate the priors using the MLE.

dims = [1 2 4 8 16 24 32 40 48 56 64]; % Desired dimensions.
[~,n_dims] = size(dims);
n_class = 8; % C = 8.
n_mix = 5; % Part 1 is to train five mixtures.

mu_fg = zeros(n_mix,M*n_class);
sigma_fg = zeros(n_mix,M*n_class);
pi_fg = zeros(n_mix,n_class);
mu_bg = zeros(n_mix,M*n_class);
sigma_bg = zeros(n_mix,M*n_class);
pi_bg = zeros(n_mix,n_class);
%% Training.
fprintf('Loading completed...starting training.\n');
tic;
for idx_mix = 1:n_mix
    % idx_mix % For debugging purposes. Comment later.
    [mu_fg(idx_mix,:),sigma_fg(idx_mix,:),pi_fg(idx_mix,:)] = em_calc(TrainsampleDCT_FG,n_class,M,200);
    [mu_bg(idx_mix,:),sigma_bg(idx_mix,:),pi_bg(idx_mix,:)] = em_calc(TrainsampleDCT_BG,n_class,M,200);
end
toc;
fprintf('\nTraining completed...starting prediction.');
%% Prediction and error analysis.
% To save on time, calculate the DFT coefficients of the image before-hand.
%dct_coeffs = zeros(rows_cheetah_ori*cols_cheetah_ori,M);
p_err_25 = zeros(n_mix*n_mix,n_dims);
%A_25 = zeros(rows_cheetah_ori,cols_cheetah_ori,n_mix*n_mix);
% 
% idx_track = 1;
% for idx_x = 5:rows_cheetah_pad-6
%     for idx_y = 5:cols_cheetah_pad-5
%         temp_block = dct2(cheetah_pad(idx_x-4:idx_x+3,idx_y-4:idx_y+3));
%         v(zig(:)) = temp_block(:);
%         dct_coeffs(idx_track,:) = v;
%         idx_track = idx_track+1;
%     end
% end
dct_coeffs = get_dct(cheetah_ori,zig);

% Perform the BDR prediction.
tic;
for idx_mix1 = 1:5
    for idx_mix2 = 1:5
        for idx_dim = 1:n_dims
            A_25 = bdr_predict(dct_coeffs,dims(idx_dim),...
                mu_fg(idx_mix1,:),mu_bg(idx_mix2,:),sigma_fg(idx_mix1,:),sigma_bg(idx_mix2,:),...
                pi_fg(idx_mix1,:),pi_bg(idx_mix2,:),p_fg,p_bg,rows_cheetah_ori,cols_cheetah_ori,n_class);
            p_err_25((idx_mix1-1)*5+idx_mix2,idx_dim) = calc_error(A_25,p_fg,p_bg);
        end
    end
end
toc;
% Plot the results from the 25 different classifiers.
% 12/05/2018 : Leave the plotting until you have all of the data from the
% training and predicting. 
