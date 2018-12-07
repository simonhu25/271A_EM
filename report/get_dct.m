% Written by Jun Hao Hu, University of California San Diego.
% All rights reserved.

%---------------------------------------------------------------------
% MATLAB routine that grabs the DCT coefficients of the image.
%---------------------------------------------------------------------

function dct_coeffs = get_dct(cheetah_img,zig)

[cheetah_rows,cheetah_cols] = size(cheetah_img);
M = 64;
dct_coeffs = zeros(cheetah_rows*cheetah_cols,M);
v = zeros(1,M);
for idx_x = 1:cheetah_rows
    for idx_y = 1:cheetah_cols        
        x = idx_x-4;
        y = idx_y-4;
        
        % Corner/edge cases
        if x < 1
            x = 1;
        end
        if y < 1
            y = 1;
        end
        if x+7>cheetah_rows
            x = cheetah_rows-7;
        end
        if y+7>cheetah_cols
            y = cheetah_cols-7;
        end
        dct_block = dct2(cheetah_img(x:x+7,y:y+7));
        v(zig(:)) = dct_block(:);
        dct_coeffs(cheetah_cols*(idx_x-1)+idx_y,:) = v;
    end
end
end