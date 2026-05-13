%% Time Domain Processing Function of MVDR
function out = MVDR_Zoom_Time_Domain(data, mic_pos, focus_point, fs, c)
    [N, M] = size(data);
    % 1. Time-Alignment (Pre-steering)
    % This brings the target signal into phase across all channels
    dist = sqrt(sum((mic_pos - focus_point).^2, 2));
    delays = dist / c;
    t = (0:N-1)' / fs;
    aligned_data = zeros(N, M);
    for m = 1:M
        % Using high-precision interpolation for alignment (same as your first code)
        aligned_data(:, m) = interp1(t, data(:, m), t + delays(m), 'linear', 0);
    end
    
    % 2. Adaptive Weight Calculation (Capon logic in time domain)
    % We use a block-based approach to estimate the spatial covariance matrix
    block_size = min(N, round(0.05 * fs)); % 50ms blocks
    num_blocks = floor(N / block_size);
    out = zeros(N, 1);
    
    % Constraint vector for MVDR: we want sum of weights = 1 for aligned signals
    e = ones(M, 1);
    dl_factor = 0.01; % Diagonal loading for stability
    
    for b = 1:num_blocks
        idx = (b-1)*block_size + 1 : b*block_size;
        X_block = aligned_data(idx, :); % [BlockSize x M]
    
        % Estimate Spatial Covariance Matrix R
        R = (X_k_transpose_X(X_block)) / block_size;
    
        % Regularization
        R = R + dl_factor * trace(R) * eye(M) / M + 1e-10 * eye(M);
    
        % MVDR Weights: w = (R^-1 * e) / (e' * R^-1 * e)
        % This minimizes variance (noise) while keeping target signal gain = 1
        R_inv_e = R \ e;
        w = R_inv_e / (e' * R_inv_e);
    
        % Apply weights to the block
        out(idx) = X_block * w;
    end
    
    % Handle remaining samples
    if N > num_blocks * block_size
        remaining_idx = num_blocks * block_size + 1 : N;
        out(remaining_idx) = mean(aligned_data(remaining_idx, :), 2);
    end
    
    % Final Normalization
    if max(abs(out)) > 0
        out = out / max(abs(out));
    else
        % Absolute fallback to simple sum if anything goes wrong
        out = mean(aligned_data, 2);
        out = out / (max(abs(out)) + 1e-9);
    end
end

function R = X_k_transpose_X(X)
    % Helper to compute X' * X safely
    R = X' * X;
end